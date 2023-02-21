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
   [uncomplicate.clojure-cpp :refer [pointer-seq fill! capacity! byte-buffer float-pointer double-pointer
                                     long-pointer int-pointer short-pointer byte-pointer
                                     element-count]]
   [uncomplicate.neanderthal
    [core :refer [transfer! copy! subvector vctr ge]]
    [real :refer [entry entry!]]
    [math :refer [ceil]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [printing :refer [print-vector print-ge print-uplo print-banded print-diagonal]]
    [navigation :refer :all]])
  (:import [java.nio Buffer ByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL]
           org.bytedeco.mkl.global.mkl_rt
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer BytePointer]
           [uncomplicate.neanderthal.internal.api
            VectorSpace Vector RealVector Matrix IntegerVector DataAccessor RealChangeable IntegerChangeable
            RealNativeMatrix RealNativeVector IntegerNativeVector DenseStorage FullStorage
            LayoutNavigator RealLayoutNavigator Region MatrixImplementation GEMatrix UploMatrix
            BandedMatrix PackedMatrix DiagonalMatrix]))

(declare real-block-vector)

;; ================ from buffer-block ====================================
(defn ^:private vector-seq [^Vector vector ^long i]
  (lazy-seq
   (if (< -1 i (.dim vector))
     (cons (.boxedEntry vector i) (vector-seq vector (inc i)))
     '())))
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


(deftype RealNativeCppVector [fact ^DataAccessor da eng master buf-ptr
                              ^long n ^long ofst ^long strd]
  Releaseable
  (release [_]
    (if master (release buf-ptr) true))
  Seqable
  (seq [x]
    (vector-seq x 0))
  Container
  (raw [_]
    (real-block-vector fact n))
  (raw [_ fact]
    (create-vector (factory fact) n false))
  (zero [_]
    (real-block-vector fact n))
  (zero [_ fact]
    (create-vector (factory fact) n true))
  (host [x]
    (let-release [res (raw x)]
      (copy eng x res)))
  (native [x]
    x)
  Viewable
  (view [x]
    (real-block-vector fact false buf-ptr n ofst strd))
  DenseContainer
  (view-vctr [x]
    x)
  (view-vctr [_ stride-mult]
    (real-block-vector fact false buf-ptr (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
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
  MemoryContext
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^VectorSpace y)))
  (device [_]
    :cpu)
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
    da)
  IFn$LDD
  (invokePrim [x i v]
    (.set x i v))
  IFn$LD
  (invokePrim [x i]
    (entry x i))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.set x i v))
  (invoke [x i]
    (entry x i))
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
    (.set ^RealAccessor da buf-ptr (+ ofst (* strd i)) val)
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
    ofst)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
  (dim [_]
    n)
  (entry [_ i]
    (.get ^RealAccessor da buf-ptr (+ ofst (* strd i))))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (real-block-vector fact false buf-ptr l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (real-block-vector fact 0))
  Applicative
  (pure [_ v]
    (let-release [res (real-block-vector fact 1)]
      (.set ^RealChangeable res 0 v)))
  (pure [_ v vs]
    (vctr fact (cons v vs))))

(defn real-block-vector
  ([fact master buf-ptr n ofst strd]
   (let [da (data-accessor fact)]
     (if (and (<= 0 n (.count da buf-ptr)))
       (->RealNativeCppVector fact da (vector-engine fact) master buf-ptr n ofst strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf-ptr)})))))
  ([fact n]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) n)]
     (real-block-vector
      fact true buf-ptr n 0 1))))
