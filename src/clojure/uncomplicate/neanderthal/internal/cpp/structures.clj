;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.structures
  (:require
   [uncomplicate.commons
    [core :refer [Releaseable release let-release Info info double-fn
                  wrap-float wrap-double wrap-int wrap-long wrap-short wrap-byte
                  Viewable view]]
    [utils :refer [dragan-says-ex]]]
   [uncomplicate.fluokitten.protocols
    :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative fold]]
   [uncomplicate.clojure-cpp :refer [pointer pointer-seq fill! capacity! byte-buffer float-pointer
                                     double-pointer long-pointer int-pointer short-pointer byte-pointer
                                     element-count]]
   [uncomplicate.neanderthal
    [core :refer [transfer! copy! dim subvector vctr ge matrix-type mrows ncols]]
    [math :refer [ceil]]
    [block :refer [entry-type offset stride buffer column?]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [printing :refer [print-vector print-ge print-uplo print-banded print-diagonal]]
    [common :refer [dense-rows dense-cols dense-dias require-trf real-accessor integer-accessor]]
    [navigation :refer :all]]
   [uncomplicate.neanderthal.internal.host.fluokitten :refer :all])
  (:import [java.nio Buffer ByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL IFn$LLLL]
           org.bytedeco.mkl.global.mkl_rt
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer
            BytePointer]
           [uncomplicate.neanderthal.internal.api Block
            VectorSpace Vector RealVector Matrix IntegerVector DataAccessor RealChangeable
            IntegerChangeable RealNativeMatrix IntegerNativeMatrix RealNativeVector
            IntegerNativeVector DenseStorage FullStorage LayoutNavigator Region
            MatrixImplementation GEMatrix UploMatrix BandedMatrix PackedMatrix DiagonalMatrix
            RealAccessor IntegerAccessor]))

(declare real-block-vector integer-block-vector cs-vector integer-ge-matrix real-ge-matrix)

;; ================ Pointer data accessors  ====================================

(defmacro put* [pt p i a]
  `(. ~(with-meta p {:tag pt}) put (long ~i) ~a))

(defmacro get* [pt p i]
  `(. ~(with-meta p {:tag pt}) get (long ~i)))

(defprotocol Destructor
  (destruct [this p]))

(defmacro def-accessor-type [name accessor-interface pointer-class entry-class pointer wrap-fn cast cast-get]
  `(deftype ~name [construct# destruct#]
     DataAccessor
     (entryType [_#]
       (. ~entry-class TYPE))
     (entryWidth [_#]
       (. ~entry-class BYTES))
     (count [_# p#]
       (element-count p#))
     (createDataSource [_# n#]
       (capacity! (~pointer (construct# (* (. ~entry-class BYTES) (max 1 n#)))) n#))
     (initialize [_# p#]
       (fill! p# 0))
     (initialize [_# p# v#]
       (fill! p# v#))
     (wrapPrim [_# v#]
       (~wrap-fn v#))
     (castPrim [_# v#]
       (~cast v#))
     DataAccessorProvider
     (data-accessor [this#]
       this#)
     Destructor
     (destruct [_# p#]
       (destruct# p#))
     MemoryContext
     (compatible? [this# o#]
       (let [da# (data-accessor o#)]
         (or (identical? this# da#) (instance? ~name da#))))
     ~accessor-interface
     (get [_# p# i#]
       (~cast-get (get* ~pointer-class p# i#)))
     (set [_# p# i# val#]
       (put* ~pointer-class p# i# val#)
       p#)))

(def-accessor-type DoublePointerAccessor RealAccessor DoublePointer Double double-pointer wrap-double double double)
(def-accessor-type FloatPointerAccessor RealAccessor FloatPointer Float float-pointer wrap-float float float)
(def-accessor-type LongPointerAccessor IntegerAccessor LongPointer Long long-pointer wrap-long long long)
(def-accessor-type IntPointerAccessor IntegerAccessor IntPointer Integer int-pointer wrap-int int int)
(def-accessor-type ShortPointerAccessor IntegerAccessor ShortPointer Short short-pointer wrap-short short long)
(def-accessor-type BytePointerAccessor IntegerAccessor BytePointer Byte byte-pointer wrap-byte byte long)

;; =======================================================================

;; ================ from buffer-block ====================================
(defn vector-seq [^Vector vector ^long i]
  (lazy-seq
   (if (< -1 i (.dim vector))
     (cons (.boxedEntry vector i) (vector-seq vector (inc i)))
     '())))

(defmacro ^:private transfer-vector-vector [source destination]
  `(do
     (if (compatible? ~source ~destination)
       (when-not (identical? ~source ~destination)
         (let [n# (min (.dim ~source) (.dim ~destination))]
           (subcopy (engine ~source) ~source ~destination 0 n# 0)))
       (dotimes [i# (min (.dim ~source) (.dim ~destination))]
         (.set ~destination i# (.entry ~source i#))))
     ~destination))

(defmacro ^:private transfer-vector-array [source destination]
  `(let [n# (min (.dim ~source) (alength ~destination))]
     (dotimes [i# n#]
       (aset ~destination i# (.entry ~source i#)))
     ~destination))

(defmacro ^:private transfer-array-vector [source destination]
  `(let [n# (min (alength ~source) (.dim ~destination))]
     (dotimes [i# n#]
       (.set ~destination i# (aget ~source i#)))
     ~destination))

(defmacro ^:private transfer-seq-vector [source destination]
  `(let [n# (.dim ~destination)]
     (loop [i# 0 src# (seq ~source)]
       (when (and src# (< i# n#))
         (.set ~destination i# (first src#))
         (recur (inc i#) (next src#))))
     ~destination))

(defmacro ^:private transfer-seq-matrix [typed-accessor source destination]
  `(if (sequential? (first ~source))
     (let [n# (.fd (storage ~destination))
           nav# (navigator ~destination)
           transfer-method# (get-method transfer!
                                        [(type (first ~source))
                                         (type (.stripe nav# ~destination 0))])]
       (loop [i# 0 s# ~source]
         (if (and s# (< i# n#))
           (do
             (transfer-method# (first s#) (.stripe nav# ~destination i#))
             (recur (inc i#) (next s#)))
           ~destination)))
     (let [da# (~typed-accessor ~destination)
           buf# (.buffer ~destination)
           ofst# (.offset ~destination)]
       (doseq-layout ~destination i# j# idx# ~source e# (.set da# buf# (+ ofst# idx#) e#))
       ~destination)))

(defmacro ^:private transfer-matrix-matrix
  ([typed-accessor condition source destination]
   `(do
      (if (and (<= (.mrows ~destination) (.mrows ~source)) (<= (.ncols ~destination) (.ncols ~source)))
        (if (and (compatible? ~source ~destination) (fits? ~source ~destination) ~condition)
          (copy (engine ~source) ~source ~destination)
          (let [flipper# (real-flipper (navigator ~destination))
                da# (~typed-accessor ~destination)
                buf# (.buffer ~destination)
                ofst# (.offset ~destination)]
            (doall-layout ~destination i# j# idx# (.set da# buf# (+ ofst# idx#) (.get flipper# ~source i# j#)))))
        (dragan-says-ex "There is not enough entries in the source matrix. Provide an appropriate submatrix of the destination."
                        {:source (info ~source) :destination (info ~destination)}))
      ~destination))
  ([typed-accessor source destination]
   `(transfer-matrix-matrix ~typed-accessor true ~source ~destination)))

(defmacro ^:private transfer-array-matrix [typed-accessor source destination]
  ` (let [da# (~typed-accessor ~destination)
          nav# (navigator ~destination)
          stor# (storage ~destination)
          reg# (region ~destination)
          buf# (.buffer ~destination)
          ofst# (.offset ~destination)
          len# (alength ~source)]
      (doall-layout nav# stor# reg# i# j# idx# cnt#
                    (when (< cnt# len#)
                      (.set da# buf# (+ ofst# idx#) (aget ~source cnt#))))
      ~destination))

(defmacro ^:private transfer-matrix-array [typed-accessor source destination]
  `(let [da# (~typed-accessor ~source)
         nav# (navigator ~source)
         stor# (storage ~source)
         reg# (region ~source)
         buf# (.buffer ~source)
         ofst# (.offset ~source)
         len# (alength ~destination)]
     (doall-layout nav# stor# reg# i# j# idx# cnt#
                   (when (< cnt# len#)
                     (aset ~destination cnt# (.get da# buf# (+ ofst# idx#)))))
     ~destination))

(defmacro ^:private transfer-vector-matrix [typed-accessor source destination]
  `(let [stor# (storage ~destination)]
     (if (and (compatible? ~source ~destination) (.isGapless stor#))
       (let [dst-view# ^VectorSpace (view-vctr ~destination)
             n# (min (.dim ~source) (.dim dst-view#))]
         (when (pos? n#)
           (subcopy (engine ~source) ~source dst-view# 0 n# 0)))
       (let [da# (~typed-accessor ~destination)
             nav# (navigator ~destination)
             reg# (region ~destination)
             buf# (.buffer ~destination)
             ofst# (.offset ~destination)
             dim# (.dim ~destination)]
         (doall-layout nav# stor# reg# i# j# idx# cnt#
                       (when (< cnt# dim#)
                         (.set da# buf# (+ ofst# idx#) (.entry ~source cnt#))))))
     ~destination))

(defmacro ^:private transfer-matrix-vector [typed-accessor source destination]
  `(let [stor# (storage ~source)]
     (if (and (compatible? ~destination ~source) (.isGapless stor#))
       (let [src-view# ^VectorSpace (view-vctr ~source)
             n# (min (.dim src-view#) (.dim ~destination))]
         (when (pos? n#)
           (subcopy (engine src-view#) src-view# ~destination 0 n# 0)))
       (let [da# (~typed-accessor ~source)
             nav# (navigator ~source)
             reg# (region ~source)
             buf# (.buffer ~source)
             ofst# (.offset ~source)
             dim# (.dim ~destination)]
         (doall-layout nav# stor# reg# i# j# idx# cnt#
                       (when (< cnt# dim#)
                         (.set ~destination cnt# (.get da# buf# (+ ofst# idx#)))))))
     ~destination))

(defmacro matrix-alter [ifn-oo ifn-lloo f nav stor reg da buf]
  `(if (instance? ~ifn-oo ~f)
     (doall-layout ~nav ~stor ~reg i# j# idx#
                   (.set ~da ~buf idx# (.invokePrim ~(with-meta f {:tag ifn-oo}) (.get ~da ~buf idx#))))
     (if (.isRowMajor ~nav)
       (doall-layout ~nav ~stor ~reg i# j# idx#
                     (.set ~da ~buf idx# (.invokePrim ~(with-meta f {:tag ifn-lloo})
                                                      j# i# (.get ~da ~buf idx#))))
       (doall-layout ~nav ~stor ~reg i# j# idx#
                     (.set ~da ~buf idx# (.invokePrim ~(with-meta f {:tag ifn-lloo})
                                                      i# j# (.get ~da ~buf idx#)))))))

;; =======================================================================

(defn block-vector
  ([constructor fact master buf-ptr n ofst strd]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 n (.count da buf-ptr))
       (constructor fact da (vector-engine fact) master (pointer buf-ptr ofst) n strd)
       (throw (ex-info "Insufficient buffer size." {:n n :offset ofst :buffer-size (.count da buf-ptr)})))))
  ([constructor fact master buf-ptr n strd]
   (block-vector constructor fact master buf-ptr n 0 strd))
  ([constructor fact n strd]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) n)]
     (block-vector constructor fact true buf-ptr n 0 strd)))
  ([constructor fact n]
   (block-vector constructor fact n 1)))

;; TODO move to general namespace
(defmacro extend-base [name]
  `(extend-type ~name
     Releaseable
     (release [this#]
       (if (.-master this#)
         (if (destruct (data-accessor this#) (buffer this#))
           true
           false)
         true))
     EngineProvider
     (engine [this#]
       (.-eng this#))
     FactoryProvider
     (factory [this#]
       (.-fact this#))
     (native-factory [this#]
       (native-factory (.-fact this#)))
     (index-factory [this#]
       (index-factory (.-fact this#)))
     DataAccessorProvider
     (data-accessor [this#]
       (.-da this#))))

;; TODO extract general cpu/gpu parts to a more general macro
(defmacro extend-block-vector [name block-vector ge-matrix]
  `(extend-type ~name
     Info
     (info
       ([this#]
        {:entry-type (.entryType (data-accessor this#))
         :class ~name
         :device :cpu
         :dim (dim this#)
         :offset (offset this#)
         :stride (.-strd this#)
         :master (.-master this#)
         :engine (info (.-eng this#))})
       ([this# info-type#]
        (case info-type#
          :entry-type (.entryType (data-accessor this#))
          :class ~name
          :device :cpu
          :dim (dim this#)
          :offset (offset this#)
          :stride (.-strd this#)
          :master (.-master this#)
          :engine (info (.-eng this#))
          nil)))
     Container
     (raw
       ([this#]
        (~block-vector (.-fact this#) (.-n this#)))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) false)))
     (zero
       ([this#]
        (create-vector (.-fact this#) (.-n this#) true))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~block-vector (.-fact this#) false (.-buf-ptr this#) (.-n this#) 0 (.-strd this#)))
     DenseContainer
     (view-vctr
       ([this#]
        this#)
       ([this# stride-mult#]
        (~block-vector (.-fact this#) false (.-buf-ptr this#)
         (ceil (/ (.-n this#) (long stride-mult#))) 0 (* (long stride-mult#) (.-strd this#)))))
     (view-ge
       ([this#]
        (~ge-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) 1 0
         (layout-navigator true) (full-storage true (.-n this#) 1) (ge-region (.-n this#) 1)))
       ([this# stride-mult#]
        (view-ge (view-ge this#) stride-mult#))
       ([this# m# n#]
        (view-ge (view-ge this#) m# n#)))
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# y#]
       (= (.-n this#) (dim y#)))
     (device [_#]
       :cpu) ;; TODO Perhaps move this to factory?
     Monoid
     (id [this#]
       (~block-vector (.-fact this#) 0))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~block-vector (.-fact this#) 1)]
          (uncomplicate.neanderthal.core/entry! res# 0 v#)))
       ([this# v# vs#]
        (vctr (.-fact this#) (cons v# vs#))))))

(defmacro extend-vector-fluokitten [t cast indexed-fn]
  `(extend ~t
     Functor
     {:fmap (vector-fmap ~t ~cast)}
     PseudoFunctor
     {:fmap! (vector-fmap identity ~t ~cast)}
     Foldable
     {:fold (vector-fold ~t ~cast ~cast)
      :foldmap (vector-foldmap ~t ~cast ~cast ~indexed-fn)}
     Magma
     {:op (constantly vector-op)}))

;; ============ Integer Vector ====================================================

(deftype IntegerBlockVector [fact ^IntegerAccessor da eng master buf-ptr
                             ^long n ^long strd]
  Object
  (hashCode [x]
    (-> (hash :IntegerBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? IntegerBlockVector y) (compatible? da y) (fits? x y))
      (or (= buf-ptr (.buffer ^Block y))
          (loop [i 0]
            (if (< i n)
              (and (= (.entry x i) (.entry ^IntegerBlockVector y i)) (recur (inc i)))
              true)))
      :default false))
  (toString [_]
    (format "#IntegerBlockVector[%s, n:%d, stride:%d]" (entry-type da) n strd))
  Seqable
  (seq [x]
    (vector-seq x 0))
  IFn$LLL
  (invokePrim [x i v]
    (if (< -1 i n)
      (.set x i v)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$LL
  (invokePrim [x i]
    (if (< -1 i n)
      (.entry x i)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.invokePrim x i v))
  (invoke [x i]
    (.invokePrim x i))
  (invoke [x]
    n)
  IntegerChangeable
  (isAllowed [x i]
    (< -1 i n))
  (set [x val]
    (set-all eng val x)
    x)
  (set [x i val]
    (.set da buf-ptr (* strd i) val)
    x)
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [x f]
    (if (instance? IFn$LL f)
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$LL f (.entry x i))))
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$LLL f i (.entry x i)))))
    x)
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$LL f (.entry x i))))
  IntegerNativeVector
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    strd)
  (isContiguous [_]
    (or (= 1 strd) (= 0 strd)))
  (dim [_]
    n)
  (entry [_ i]
    (.get da buf-ptr (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (integer-block-vector fact false buf-ptr l (* k strd) strd)))

(extend-base IntegerBlockVector)
(extend-block-vector IntegerBlockVector integer-block-vector integer-ge-matrix)
(extend-vector-fluokitten IntegerBlockVector long IFn$LLL)

(def integer-block-vector (partial block-vector ->IntegerBlockVector))

(defmethod print-method IntegerBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (seq x)))))

(defmethod transfer! [IntegerBlockVector IntegerBlockVector]
  [^IntegerBlockVector source ^IntegerBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [clojure.lang.Sequential IntegerBlockVector]
  [source ^IntegerBlockVector destination]
  (transfer-seq-vector source destination))

(defmethod transfer! [(Class/forName "[D") IntegerBlockVector]
  [^doubles source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[F") IntegerBlockVector]
  [^floats source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[J") IntegerBlockVector]
  [^longs source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[I") IntegerBlockVector]
  [^ints source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[S") IntegerBlockVector]
  [^longs source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[B") IntegerBlockVector]
  [^ints source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[J")]
  [^IntegerBlockVector source ^longs destination]
  (transfer-vector-array source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[I")]
  [^IntegerBlockVector source ^ints destination]
  (transfer-vector-array source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[S")]
  [^IntegerBlockVector source ^longs destination]
  (transfer-vector-array source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[B")]
  [^IntegerBlockVector source ^ints destination]
  (transfer-vector-array source destination))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [fact ^RealAccessor da eng master buf-ptr
                          ^long n ^long strd]
  Object
  (hashCode [x]
    (-> (hash :RealBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? RealBlockVector y) (compatible? da y) (fits? x y))
      (or (= buf-ptr (.buffer ^Block y))
          (loop [i 0]
            (if (< i n)
              (and (= (.entry x i) (.entry ^RealBlockVector y i)) (recur (inc i)))
              true)))
      :default false))
  (toString [_]
    (format "#RealBlockVector[%s, n:%d, stride:%d]" (entry-type da) n strd))
  Seqable
  (seq [x]
    (vector-seq x 0))
  IFn$LDD
  (invokePrim [x i v]
    (if (< -1 i n)
      (.set x i v)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$LD
  (invokePrim [x i]
    (if (< -1 i n)
      (.entry x i)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.invokePrim x i v))
  (invoke [x i]
    (.invokePrim x i))
  (invoke [x]
    n)
  RealChangeable
  (set [x val]
    (if-not (Double/isNaN val)
      (set-all eng val x)
      (dotimes [i n]
        (.set x i val)))
    x)
  (set [x i val]
    (.set da buf-ptr (* strd i) val)
    x)
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [x f]
    (if (instance? IFn$DD f)
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$LDD f i (.entry x i)))))
    x)
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealNativeVector
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
  (dim [_]
    n)
  (entry [_ i]
    (.get da buf-ptr (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (real-block-vector fact false buf-ptr l (* k strd) strd)))

(extend-base RealBlockVector)
(extend-block-vector RealBlockVector real-block-vector real-ge-matrix)
(extend-vector-fluokitten RealBlockVector double IFn$LDD)

(def real-block-vector (partial block-vector ->RealBlockVector))

(defmethod print-method RealBlockVector [^Vector x ^java.io.Writer w]
  (.write w (str x))
  (print-vector w x))

(defmethod transfer! [RealBlockVector RealBlockVector]
  [^RealBlockVector source ^RealBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [IntegerBlockVector RealBlockVector]
  [^IntegerBlockVector source ^RealBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [RealBlockVector IntegerBlockVector]
  [^RealBlockVector source ^IntegerBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [clojure.lang.Sequential RealBlockVector]
  [source ^RealBlockVector destination]
  (transfer-seq-vector source destination))

(defmethod transfer! [(Class/forName "[D") RealBlockVector]
  [^doubles source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[F") RealBlockVector]
  [^floats source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[J") RealBlockVector]
  [^longs source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[I") RealBlockVector]
  [^ints source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[D")]
  [^RealBlockVector source ^doubles destination]
  (transfer-vector-array source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[F")]
  [^RealBlockVector source ^floats destination]
  (transfer-vector-array source destination))


;; ======================= Compressed Sparse Vector ======================================
;; TODO move to API
(defprotocol SparseBlas
  (gthr [this y x]))

;; TODO Move to a more general namespace.

(defprotocol CompressedSparse
  (entries [this])
  (indices [this]))

(defprotocol CSR
  (columns [this])
  (indexb [this])
  (indexe [this]))

(defprotocol SparseFactory ;;TODO move to api.
  (create-ge-csr [this m n ind ind-b ind-e column? init])
  (create-tr-csr [this m n ind ind-b ind-e column? lower? diag-unit? init])
  (create-sy-csr [this m n ind ind-b ind-e column? lower? diag-unit? init])
  (cs-vector-engine [this])
  (csr-engine [this]))

(deftype CSVector [fact eng ^Block nzx ^IntegerVector indx ^long n]
  Object
  (hashCode [_]
    (-> (hash :CSVector) (hash-combine nzx) (hash-combine indx)))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (instance? CSVector y)
      (and  (= nzx (entries y)) (= indx (indices y)))
      :default :false))
  (toString [_]
    (format "#CSVector[%s, n:%d, nnz:%d]" (entry-type (data-accessor nzx)) n (dim nzx)))
  Info
  (info [x]
    {:entry-type (.entryType (data-accessor nzx))
     :class (class x)
     :device (info nzx :device)
     :dim n
     :engine (info eng)})
  (info [x info-type]
    (case info-type
      :entry-type (.entryType (data-accessor nzx))
      :class (class x)
      :device (info nzx :device)
      :dim n
      :engine (info eng)
      nil))
  Releaseable
  (release [_]
    (release nzx)
    (release indx)
    true)
  Seqable
  (seq [x]
    (vector-seq nzx 0))
  MemoryContext
  (compatible? [_ y]
    (compatible? nzx (entries y)))
  (fits? [_ y]
    (and (= n (dim y))
         (or (nil? (indices y)) (= indx (indices y)))))
  (device [_]
    (device nzx))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor fact))
  Container
  (raw [x]
    (raw x fact))
  (raw [_ fact]
    (cs-vector (factory fact) n (view indx) false))
  (zero [x]
    (zero x fact))
  (zero [_ fact]
    (cs-vector (factory fact) n (view indx) true))
  (host [x]
    (let-release [host-nzx (host nzx)
                  host-indx (host indx)]
      (cs-vector n host-indx host-nzx)))
  (native [x]
    (let-release [native-nzx (native nzx)]
      (if (= nzx native-nzx)
        x
        (let-release [native-indx (native indx)]
          (cs-vector n native-indx native-nzx)))))
  Viewable
  (view [x]
    (cs-vector fact (view nzx) (view indx)))
  Block
  (buffer [_]
    (.buffer nzx))
  (offset [_]
    (.offset nzx))
  (stride [_]
    (.stride nzx))
  (isContiguous [_]
    (.isContiguous ^Block nzx))
  VectorSpace
  (dim [_]
    n)
  ;; TODO perhaps implement Vector?
  CompressedSparse
  (entries [_]
    nzx)
  (indices [_]
    indx))

(extend-type RealBlockVector
  CompressedSparse
  (entries [this]
    this)
  (indices [this]
    nil))

(defn cs-vector
  ([^long n indx nzx]
   (let [fact (factory nzx)]
     (if (= (factory indx) (index-factory nzx))
       (if (and (<= 0 (dim nzx) n) (<= 0 (dim indx)) (fits? nzx indx)
                (= 1 (stride indx) (stride nzx)) (= 0 (offset indx) (offset nzx)))
         (->CSVector fact (cs-vector-engine fact) nzx indx n)
         (throw (ex-info "Non-zero vector and index vector have to fit each other." {:nzx nzx :indx indx})));;TODO error message
       (throw (ex-info "Incompatible index vector" {:required (index-factory nzx) :provided (factory indx)})))))
  ([fact ^long n indx init]
   (let-release [nzx (create-vector fact (dim indx) init)]
     (cs-vector n indx nzx))))

(defmethod print-method CSVector [^Vector x ^java.io.Writer w]
  (.write w (format "%s\n%s" (str x) (pr-str (seq (indices x)))))
  (print-vector w (entries x)))

(defmethod transfer! [CSVector CSVector]
  [source destination]
  (transfer! (entries source) (entries destination))
  destination)

(defn transfer-seq-csvector [source destination]
  (let [[s0 s1] source]
    (if (number? s0)
      (transfer! source (entries destination))
      (do
        (transfer! s0 (indices destination))
        (if (sequential? s1)
          (transfer! s1 (entries destination))
          (transfer! (rest source) (entries destination)))))))

(defmethod transfer! [clojure.lang.Sequential CSVector]
  [source destination]
  (transfer-seq-csvector source destination)
  destination)

(defmethod transfer! [(Class/forName "[D") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[F") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[J") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[I") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [CSVector (Class/forName "[D")]
  [source destination]
  (transfer! (entries source) destination))

(defmethod transfer! [CSVector (Class/forName "[F")]
  [source destination]
  (transfer! (entries source) destination))

;;TODO handle heterogenous types (float/double...)
(defmethod transfer! [RealBlockVector CSVector]
  [^RealBlockVector source ^CSVector destination]
  (gthr (engine destination) source destination)
  destination)

;; =================== Matrices ================================================

(defmacro extend-ge-matrix [name block-vector ge-matrix]
  `(extend-type ~name
     Info
     (info
       ([this#]
        {:entry-type (.entryType (data-accessor this#))
         :class ~name
         :device :cpu
         :matrix-type :ge
         :dim (dim this#)
         :m (.-m this#)
         :n (.-n this#)
         :offset (offset this#)
         :stride (stride this#)
         :master (.-master this#)
         :layout (:layout (info (.-nav this#)))
         :storage (info (.-nav this#))
         :region (info (.-reg this#))
         :engine (info (.-eng this#))})
       ([this# info-type#]
        (case info-type#
          :entry-type (.entryType (data-accessor this#))
          :class ~name
          :device :cpu
          :matrix-type :ge
          :dim (dim this#)
          :m (.-m this#)
          :n (.-n this#)
          :offset (offset this#)
          :stride (stride this#)
          :master (.-master this#)
          :layout (:layout (info (.-nav this#)))
          :storage (info (.-nav this#))
          :region (info (.-reg this#))
          :engine (info (.-eng this#))
          nil)))
     Navigable
     (navigator [this#]
       (.-nav this#))
     (storage [this#]
       (.-stor this#))
     (region [this#]
       (.-reg this#))
     Container
     (raw
       ([this#]
        (~ge-matrix (.-fact this#) (.-m this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)))
       ([this# fact#]
        (create-ge (factory fact#) (.-m this#) (.-n this#) (column? (.-nav this#)) false)))
     (zero
       ([this#]
        (create-ge (.-fact this#) (.-m this#) (.-n this#) (column? (.-nav this#)) true))
       ([this# fact#]
        (create-ge (factory fact#) (.-m this#) (.-n this#) (column? (.-nav this#)) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)
         res#))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~ge-matrix (.-fact this#) false (.-buf-ptr this#)
        (.-m this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)))
     DenseContainer
     (view-vctr
       ([this#]
        (if (.isContiguous this#)
          (~block-vector (.-fact this#) false (.-buf-ptr this#) (.dim this#) 0 1)
          (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:a (info this#)}))))
       ([this# stride-mult#]
        (view-vctr (view-vctr this#) stride-mult#)))
     (view-ge
       ([this#]
        this#)
       ([this# stride-mult#]
        (let [shrinked# (ceil (/ (.invokePrim this#) (long stride-mult#)))
              column-major# (column? (.-nav this#))
              [m# n#] (if column-major# [(.m this#) shrinked#] [shrinked# (.-n this#)])]
          (~ge-matrix (.-fact this#) false (.-buf-ptr this#) m# n# (.-nav this#)
           (full-storage column-major# m# n# (* (long stride-mult#) (stride this#)))
           (ge-region m# n#))))
       ([this# m# n#]
        (if (.isContiguous this#)
          (~ge-matrix (.-fact this#) false (.-buf-ptr this#) m# n# (.-nav this#)
           (full-storage (column? (.-nav this#)) m# n#) (ge-region m# n#))
          (throw (ex-info "Strided GE matrix cannot be viewed through different dimensions." {:a (info this#)})))))
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# b#]
       (and (instance? GEMatrix b#) (= (.-reg this#) (region b#))))
     (device [this#]
       :cpu)
     Monoid
     (id [this#]
       (~ge-matrix (.-fact this#) 0 0 (column? (.-nav this#))))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~ge-matrix (.-fact this#) 1 1 (column? (.-nav this#)))]
          (uncomplicate.neanderthal.core/entry! res# 0 0 v#)))
       ([this# v# vs#]
        (ge (.-fact this#) (cons v# vs#))))))

(defmacro extend-ge-trf [name]
  `(extend-type ~name
     Triangularizable
     (create-trf [a# pure#]
       (lu-factorization a pure))
     (create-ptrf [a#]
       (dragan-says-ex "Pivotless factorization is not available for GE matrices."))
     TRF
     (trtrs! [a# b#]
       (require-trf))
     (trtrs [a# b#]
       (require-trf))
     (trtri! [a#]
       (require-trf))
     (trtri [a#]
       (require-trf))
     (trcon
       ([a# nrm# nrm1?#]
        (require-trf))
       ([a# nrm1?#]
        (require-trf)))
     (trdet [a#]
       (require-trf))))

(defmacro extend-matrix-fluokitten [t cast typed-flipper typed-accessor]
  `(extend ~t
     Functor
     {:fmap (matrix-fmap ~typed-flipper ~typed-accessor ~cast)}
     PseudoFunctor
     {:fmap! (matrix-fmap ~typed-flipper ~typed-accessor identity ~cast)}
     Foldable
     {:fold (matrix-fold ~typed-flipper ~cast)
      :foldmap (matrix-foldmap ~typed-flipper ~cast)}
     Magma
     {:op (constantly matrix-op)}))

;; =================== Real Matrix =============================================

(defmacro matrix-equals [flipper da a b]
  `(or (identical? ~a ~b) (= (.buffer ~a) (buffer ~b))
       (and (instance? (class ~a) ~b)
            (= (.matrixType ~a) (matrix-type ~b))
            (compatible? ~a ~b) (= (.mrows ~a) (mrows ~b)) (= (.ncols ~a) (ncols ~b))
            (let [buf# (.buffer ~a)]
              (and-layout ~a i# j# idx# (= (.get ~da buf# idx#) (.get ~flipper ~b i# j#)))))))

(deftype RealGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg
                       fact ^RealAccessor da eng master buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (real-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#RealGEMatrix[%s, mxn:%dx%d, layout%s]"
            (entry-type da) m n (dec-property (.layout nav))))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 (.fd stor))))
  IFn$LLDD
  (invokePrim [a i j v]
    (if (and (< -1 i m) (-1 j n))
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i m) (-1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
  (invoke [a]
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    (and (< -1 i m) (-1 j n)))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$DD IFn$LLDD f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    (.isGapless stor))
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (.get da buf-ptr (.index nav stor i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (real-block-vector fact false buf-ptr n (.index nav stor i 0)
                       (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (real-block-vector fact false buf-ptr m (.index nav stor 0 j)
                       (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf-ptr (min m n) 0 (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (real-block-vector fact false buf-ptr (min m (- n k)) (.index nav stor 0 k) (inc (.ld stor)))
      (real-block-vector fact false buf-ptr (min (+ m k) n) (.index nav stor (- k) 0) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (real-ge-matrix fact false buf-ptr k l (.index nav stor i j)
                    nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (real-ge-matrix fact false buf-ptr n m 0 (flip nav) stor (flip reg))))

(extend-base RealGEMatrix)
(extend-ge-matrix RealGEMatrix real-block-vector real-ge-matrix)
(extend-matrix-fluokitten RealGEMatrix double real-flipper real-accessor)

(defn ge-matrix
  ([constructor fact master buf-ptr m n ofst nav ^FullStorage stor reg]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 (.capacity stor) (.count da buf-ptr))
       (constructor nav stor reg fact da (ge-engine fact) master buf-ptr m n)
       (throw (ex-info "Insufficient buffer size."
                       {:dim (.capacity stor) :buffer-size (.count da buf-ptr)})))))
  ([constructor fact m n nav ^FullStorage stor reg]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) (.capacity stor))]
     (ge-matrix constructor fact true buf-ptr m n 0 nav stor reg)))
  ([constructor fact m n column?]
   (ge-matrix constructor fact m n (layout-navigator column?) (full-storage column? m n) (ge-region m n)))
  ([constructor fact m n]
   (ge-matrix constructor fact m n true)))

(def real-ge-matrix (partial ge-matrix ->RealGEMatrix))

(defmethod transfer! [clojure.lang.Sequential RealGEMatrix]
  [source ^RealGEMatrix destination]
  (transfer-seq-matrix real-accessor source destination))

(defmethod transfer! [RealGEMatrix RealGEMatrix] ;;TODO RealNativeMatrix, once its merged with neanderthal
  [^RealGEMatrix source ^RealGEMatrix destination]
  (transfer-matrix-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[D") RealNativeMatrix]
  [^doubles source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[F") RealNativeMatrix]
  [^floats source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[J") RealNativeMatrix]
  [^longs source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[I") RealNativeMatrix]
  [^ints source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[S") RealNativeMatrix]
  [^shorts source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[B") RealNativeMatrix]
  [^bytes source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[D")]
  [^RealNativeMatrix source ^doubles destination]
  (transfer-matrix-array real-accessor source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[F")]
  [^RealNativeMatrix source ^floats destination]
  (transfer-matrix-array real-accessor source destination))

(defmethod transfer! [RealNativeVector RealGEMatrix]
  [^RealNativeVector source ^RealGEMatrix destination]
  (transfer-vector-matrix real-accessor source destination))

(defmethod transfer! [RealGEMatrix RealNativeVector]
  [^RealGEMatrix source ^RealBlockVector destination]
  (transfer-matrix-vector real-accessor source destination))

(defmethod transfer! [IntegerNativeVector RealGEMatrix]
  [^IntegerNativeVector source ^RealGEMatrix destination]
  (transfer-vector-matrix real-accessor source destination))

(defmethod transfer! [RealGEMatrix IntegerNativeVector]
  [^RealGEMatrix source ^IntegerBlockVector destination]
  (transfer-matrix-vector real-accessor source destination))

;; =================== Integer Matrix =============================================

(deftype IntegerGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg
                          fact ^IntegerAccessor da eng master
                          buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :IntegerGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (integer-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#IntegerGEMatrix[%s, mxn:%dx%d, layout%s]"
            (entry-type da) m n (dec-property (.layout nav))))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 (.fd stor))))
  IFn$LLDD
  (invokePrim [a i j v]
    (if (and (< -1 i m) (-1 j n))
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i m) (-1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
  (invoke [a]
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  IntegerChangeable
  (isAllowed [a i j]
    (and (< -1 i m) (-1 j n)))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$LL IFn$LLLL f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$LL f (.get da buf-ptr idx)))
      a))
  IntegerNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    (.isGapless stor))
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (.get da buf-ptr (.index nav stor i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (integer-block-vector fact false buf-ptr n (.index nav stor i 0)
                          (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (integer-block-vector fact false buf-ptr m (.index nav stor 0 j)
                          (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (integer-block-vector fact false buf-ptr (min m n) 0 (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (integer-block-vector fact false buf-ptr (min m (- n k)) (.index nav stor 0 k) (inc (.ld stor)))
      (integer-block-vector fact false buf-ptr (min (+ m k) n) (.index nav stor (- k) 0) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (integer-ge-matrix fact false buf-ptr k l (.index nav stor i j)
                       nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (integer-ge-matrix fact false buf-ptr n m 0 (flip nav) stor (flip reg))))

(extend-base IntegerGEMatrix)
(extend-ge-matrix IntegerGEMatrix integer-block-vector integer-ge-matrix)
(extend-matrix-fluokitten IntegerGEMatrix long integer-flipper integer-accessor)

(def integer-ge-matrix (partial ge-matrix ->IntegerGEMatrix))

(defmethod transfer! [clojure.lang.Sequential IntegerGEMatrix]
  [source ^IntegerGEMatrix destination]
  (transfer-seq-matrix integer-accessor source destination))

(defmethod transfer! [IntegerGEMatrix IntegerGEMatrix] ;;TODO IntegerNativeMatrix, once its merged with neanderthal
  [^IntegerGEMatrix source ^IntegerGEMatrix destination]
  (transfer-matrix-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[D") IntegerNativeMatrix]
  [^doubles source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[F") IntegerNativeMatrix]
  [^floats source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[J") IntegerNativeMatrix]
  [^longs source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[I") IntegerNativeMatrix]
  [^ints source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[S") IntegerNativeMatrix]
  [^shorts source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[B") IntegerNativeMatrix]
  [^bytes source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[J")]
  [^IntegerNativeMatrix source ^longs destination]
  (transfer-matrix-array integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[I")]
  [^IntegerNativeMatrix source ^ints destination]
  (transfer-matrix-array integer-accessor source destination))

(defmethod transfer! [IntegerNativeVector IntegerGEMatrix]
  [^IntegerNativeVector source ^IntegerGEMatrix destination]
  (transfer-vector-matrix integer-accessor source destination))

(defmethod transfer! [IntegerGEMatrix IntegerNativeVector]
  [^IntegerGEMatrix source ^IntegerBlockVector destination]
  (transfer-matrix-vector integer-accessor source destination))

(defmethod transfer! [IntegerNativeVector IntegerGEMatrix]
  [^IntegerNativeVector source ^IntegerGEMatrix destination]
  (transfer-vector-matrix integer-accessor source destination))

(defmethod transfer! [IntegerGEMatrix IntegerNativeVector]
  [^IntegerGEMatrix source ^IntegerBlockVector destination]
  (transfer-matrix-vector integer-accessor source destination))
