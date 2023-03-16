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
    [core :refer [transfer! copy! dim subvector vctr ge]]
    [real :refer [entry entry!]]
    [math :refer [ceil]]
    [block :refer [entry-type offset stride buffer column?]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [printing :refer [print-vector print-ge print-uplo print-banded print-diagonal]]
    [common :refer [dense-rows dense-cols dense-dias require-trf]]
    [navigation :refer :all]])
  (:import [java.nio Buffer ByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL]
           org.bytedeco.mkl.global.mkl_rt
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer
            BytePointer]
           [uncomplicate.neanderthal.internal.api Block
            VectorSpace Vector RealVector Matrix IntegerVector DataAccessor RealChangeable
            IntegerChangeable RealNativeMatrix RealNativeVector IntegerNativeVector DenseStorage
            FullStorage LayoutNavigator RealLayoutNavigator Region MatrixImplementation GEMatrix
            UploMatrix BandedMatrix PackedMatrix DiagonalMatrix]))

(declare real-block-vector integer-block-vector cs-vector real-ge-matrix)


;; ================ Pointer data accessors  ====================================

(definterface RealAccessor ;;TODO move to API. Replace RealBufferAccessor, since there's no need to specify Buffer.
  (get ^double [p ^long i])
  (set ^double [p ^long i ^double val]))

(defn real-accessor ^RealAccessor [provider]
  (data-accessor provider))

(definterface IntegerAccessor ;;TODO move to API
  (get ^long [p ^long i])
  (set ^long [p ^long i ^long val]))

(defmacro put* [pt p i a]
  `(. ~(with-meta p {:tag pt}) put (long ~i) ~a))

(defmacro get* [pt p i]
  `(. ~(with-meta p {:tag pt}) get (long ~i)))

(defmacro def-accessor-type [name accessor-interface pointer-class entry-class pointer wrap-fn cast-fn]
  `(deftype ~name [constructor#]
     DataAccessor
     (entryType [_]
       (. ~entry-class TYPE))
     (entryWidth [_]
       (. ~entry-class BYTES))
     (count [_ p#]
       (element-count p#))
     (createDataSource [_ n#]
       (capacity! (~pointer (constructor# (* (. ~entry-class BYTES) n#))) n#))
     (initialize [_ p#]
       (fill! p# 0))
     (initialize [_ p# v#]
       (fill! p# v#))
     (wrapPrim [_ v#]
       (~wrap-fn v#))
     (castPrim [_ v#]
       (~cast-fn v#))
     DataAccessorProvider
     (data-accessor [this#]
       this#)
     MemoryContext
     (compatible? [this# o#]
       (let [da# (data-accessor o#)]
         (or (identical? this# da#) (instance? ~name da#))))
     ~accessor-interface
     (get [_ p# i#]
       (get* ~pointer-class p# i#))
     (set [_ p# i# val#]
       (put* ~pointer-class p# i# val#)
       p#)))

(def-accessor-type DoublePointerAccessor RealAccessor DoublePointer Double double-pointer wrap-double double)
(def-accessor-type FloatPointerAccessor RealAccessor FloatPointer Float float-pointer wrap-float float)
(def-accessor-type LongPointerAccessor IntegerAccessor LongPointer Long long-pointer wrap-long long)
(def-accessor-type IntPointerAccessor IntegerAccessor IntPointer Integer int-pointer wrap-int int)
(def-accessor-type ShortPointerAccessor IntegerAccessor ShortPointer Short short-pointer wrap-short short)
(def-accessor-type BytePointerAccessor IntegerAccessor BytePointer Byte byte-pointer wrap-byte byte)

;; =======================================================================

;; ================ from buffer-block ====================================
(defn ^:private vector-seq [^Vector vector ^long i]
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

;; =======================================================================

(defn block-vector
  ([constructor fact master buf-ptr n ofst strd]
   (let [da (data-accessor fact)]
     (if (<= 0 n (.count da buf-ptr))
       (constructor fact da (vector-engine fact) master (pointer buf-ptr ofst) n strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf-ptr)})))))
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
       (if (.-master this#) (release (.-buf-ptr this#)) true))
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
(defmacro extend-block-vector [name block-vector]
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
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# y#]
       (= (.-n this#) (dim y#)))
     (device [_#]
       :cpu)
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
    (.set x i v))
  IFn$LL
  (invokePrim [x i]
    (.entry x i))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.set x i v))
  (invoke [x i]
    (.entry x i))
  (invoke [x]
    n)
  IntegerChangeable
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
(extend-block-vector IntegerBlockVector integer-block-vector)

(def integer-block-vector (partial block-vector ->IntegerBlockVector))

(defmethod print-method IntegerBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take *print-length* (seq x))))))

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

(defmethod transfer! [IntegerBlockVector (Class/forName "[J")]
  [^IntegerBlockVector source ^longs destination]
  (transfer-vector-array source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[I")]
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
  ;; TODO
  ;; (view-ge [_]
  ;;   (real-ge-matrix fact false buf n 1 ofst (layout-navigator true) (full-storage true n 1) (ge-region n 1)))
  ;; (view-ge [x stride-mult]
  ;;   (view-ge (view-ge x) stride-mult))
  ;; (view-ge [x m n]
  ;;   (view-ge (view-ge x) m n))
  ;; (view-tr [x uplo diag]
  ;;   (view-tr (view-ge x) uplo diag))
  ;; (view-sy [x uplo]
  ;;   (view-sy (view-ge x) uplo))
  IFn$LDD
  (invokePrim [x i v]
    (.set x i v))
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.set x i v))
  (invoke [x i]
    (.entry x i))
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
(extend-block-vector RealBlockVector real-block-vector)

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

(defprotocol SparseCompressed
  (entries [this])
  (indices [this]))

(defprotocol SparseFactory ;;TODO move to api.
  (cs-vector-engine [this]))

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
    (format "#CSVector[%s, n:%d]" (entry-type (data-accessor nzx)) n))
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
  SparseCompressed
  (entries [this]
    nzx)
  (indices [this]
    indx))

(extend-type RealBlockVector
  SparseCompressed
  (entries [this]
    this)
  (indices [this]
    nil))

(defn cs-vector
  ([^long n indx nzx]
   (let [fact (factory nzx)]
     (if (and (<= 0 (dim nzx) n) (= 1 (stride indx) (stride nzx)) (= 0 (offset indx) (offset nzx))
              (fits? nzx indx))
       (->CSVector fact (cs-vector-engine fact) nzx indx n)
       (throw (ex-info "Non-zero vector and index vector have to fit each other." {:nzx nzx :indx indx})))));;TODO error message
  ([fact ^long n indx init]
   (let-release [nzx (create-vector fact (dim indx) init)]
     (cs-vector n indx nzx))))

(defmethod print-method CSVector [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take *print-length* (indices x)))))
  (print-vector w (entries x)))

(defmethod transfer! [CSVector CSVector]
  [source destination]
  (transfer! (entries source) (entries destination)))

(defmethod transfer! [clojure.lang.Sequential CSVector]
  [source destination]
  (transfer! source (entries destination)))

(defmethod transfer! [(Class/forName "[D") CSVector]
  [source destination]
  (transfer! source (entries destination)))

(defmethod transfer! [(Class/forName "[F") CSVector]
  [source destination]
  (transfer! source (entries destination)))

(defmethod transfer! [(Class/forName "[J") CSVector]
  [source destination]
  (transfer! source (entries destination)))

(defmethod transfer! [(Class/forName "[I") CSVector]
  [source destination]
  (transfer! source (entries destination)))

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

(defmacro extend-ge-matrix [name ge-matrix]
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
         (copy (.-eng this#) this# res#)))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~ge-matrix (.-fact this#) false (.-buf-ptr this#) (.-m this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)))
     DenseContainer ;;TODO implement all view-* functions
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

;; =================== Real Matrix =============================================

(defn matrix-equals [^RealNativeMatrix a ^RealNativeMatrix b]
  (or (identical? a b) (= (.buffer a) (.buffer b))
      (and (instance? (class a) b)
           (= (.matrixType ^MatrixImplementation a) (.matrixType ^MatrixImplementation b))
           (compatible? a b) (= (.mrows a) (.mrows b)) (= (.ncols a) (.ncols b))
           (let [nav (real-navigator a)
                 da ^RealAccessor (data-accessor a)
                 buf (.buffer a)]
             (and-layout a i j idx (= (.get da buf idx) (.get nav b i j)))))))

(deftype RealGEMatrix [^RealLayoutNavigator nav ^FullStorage stor ^Region reg
                       fact ^RealAccessor da eng master
                       buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (matrix-equals a b))
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
    (entry! a i j v))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [a i j v]
    (entry! a i j v))
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    true)
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
    (if (instance? IFn$DD f)
      (doall-layout nav stor reg i j idx
                    (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx))))
      (doall-layout nav stor reg i j idx
                    (.set da buf-ptr idx (.invokePrimitive nav f i j (.get da buf-ptr idx)))))
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
(extend-ge-matrix RealGEMatrix real-ge-matrix)

(defn ge-matrix
  ([constructor fact master buf-ptr m n ofst nav ^FullStorage stor reg]
   (let [da (data-accessor fact)]
     (if (<= 0 (* (.capacity stor) (.count da buf-ptr)))
       (constructor nav stor reg fact da (ge-engine fact) master (pointer buf-ptr ofst) m n)
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
