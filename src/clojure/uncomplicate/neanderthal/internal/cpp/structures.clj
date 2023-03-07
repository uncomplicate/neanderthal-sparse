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
    [block :refer [entry-type offset stride]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [printing :refer [print-vector print-ge print-uplo print-banded print-diagonal]]
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

(declare real-block-vector integer-block-vector cs-vector)

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

;; =======================================================================

;; ================ Pointer data accessors  ====================================

(definterface RealAccessor ;;TODO move to API
  (get ^double [p ^long i])
  (set ^double [p ^long i ^double val]))

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

(defmacro extend-block-vector [name block-vector]
  `(extend-type ~name
     Info
     (info [this#]
       {:entry-type (.entryType (data-accessor this#))
        :class ~name
        :device :cpu
        :dim (dim this#)
        :offset (offset this#)
        :stride (.-strd this#)
        :master (.-master this#)
        :engine (info (.-eng this#))})
     (info [this# info-type#]
       (case info-type#
         :entry-type (.entryType (data-accessor this#))
         :class ~name
         :device :cpu
         :dim (dim this#)
         :offset (offset this#)
         :stride (.-strd this#)
         :master (.-master this#)
         :engine (info (.-eng this#))
         nil))
     Releaseable
     (release [this#]
       (if (.-master this#) (release (.-buf-ptr this#)) true))
     Container
     (raw
       ([this#]
        (~block-vector (.-fact this#) (.-n this#)))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) false)))
     (zero
       ([this#]
        (~block-vector (.-fact this#) (.-n this#)))
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
       (.-da this#))
     Monoid
     (id [this#]
       (~block-vector (.-fact this#) 0))
     Applicative
     (pure [this# v#]
       (let-release [res# (~block-vector (.-fact this#) 1)]
         (uncomplicate.neanderthal.core/entry! res# 0 v#)))
     (pure [this# v# vs#]
       (vctr (.-fact this#) (cons v# vs#)))))

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

(extend-block-vector IntegerBlockVector integer-block-vector)

(def integer-block-vector (partial block-vector ->IntegerBlockVector))

(defmethod print-method IntegerBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take 100 (seq x))))))

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
;; TODO Move to a more general namespace.

(defprotocol SparseCompressed
  (entries [this])
  (indices [this]))

(defprotocol SparseFactory ;;TODO move to api.
  (cs-vector-engine [this]))

(deftype CSVector [fact eng ^Block nzx ^Block indx ^long n]
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
  ;; Container ;;TODO
  ;; (raw [_]
  ;;   (let-release [raw-nzx (raw nzx)]
  ;;     (cs-vector fact raw-nzx (view indx))))
  ;; (raw [_ fact]
  ;;   (create-cs-vector (factory fact) n false))
  ;; (zero [x]
  ;;   (zero x fact))
  ;; (zero [_ fact]
  ;;   (create-vector (factory fact) n true))
  ;; (host [x] ;;TODO
  ;;   (let-release [res (raw x)]
  ;;     (copy eng x res)))
  ;; (native [x];;TODO
  ;;   x)
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
  ([fact ^long n indx nzx]
   (if (and (<= 0 (dim nzx) n) (= 1 (stride indx) (stride nzx)) (= 0 (offset indx) (offset nzx))
            (fits? nzx indx))
     (->CSVector fact (cs-vector-engine fact) nzx indx n)
     (throw (ex-info "Non-zero vector and index vector have to fit each other." {:nzx nzx :indx indx}))));;TODO error message
  ([fact ^long n indx]
   (let-release [nzx (create-vector fact (dim indx) false)]
     (cs-vector fact n indx nzx))))
