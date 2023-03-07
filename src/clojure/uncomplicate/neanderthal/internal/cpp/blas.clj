;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://openpsource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.blas
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release extract]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [position! pointer float-pointer double-pointer]]
            [uncomplicate.neanderthal
             [block :refer [buffer offset stride]]
             [core :refer [dim entry]]]
            [uncomplicate.neanderthal.internal.api :refer [iamax engine data-accessor]]
            [uncomplicate.neanderthal.internal.cpp.mkl.constants :refer :all])
  (:import [org.bytedeco.mkl.global mkl_rt]
           [org.bytedeco.javacpp Pointer FloatPointer DoublePointer IntPointer]
           [uncomplicate.neanderthal.internal.api VectorSpace Block RealVector]))

#_(defn cpp-pointer [constructor x]
  (position! (constructor (buffer x)) (offset x)))

;; TODO move to structures

(defn float-ptr
  (^FloatPointer [^Block x]
   (.buffer x))
  (^FloatPointer [^Block x ^long i]
   (.position (DoublePointer. ^FloatPointer (.buffer x)) i)))

(defn double-ptr
  (^DoublePointer [^Block x]
   (.buffer x))
  (^DoublePointer [^Block x ^long i]
   (.position (DoublePointer. ^DoublePointer (.buffer x)) i)))

(defn int-ptr ^IntPointer [^Block x]
  (.buffer x))

(defn coerce-double-ptr ^DoublePointer
  (^DoublePointer [^Block x]
   (DoublePointer. ^Pointer (.buffer x)))
  (^DoublePointer [^Block x ^long i]
   (.position (DoublePointer. ^Pointer (.buffer x)) i)))

(defn coerce-float-ptr
  (^FloatPointer [^Block x]
   (FloatPointer. ^Pointer (.buffer x)))
  (^FloatPointer [^Block x ^long i]
   (.position (FloatPointer. ^Pointer (.buffer x)) i)))

;; TODO Copied from host
(defn vector-imax [^RealVector x]
  (let [cnt (.dim x)]
    (if (< 0 cnt)
      (loop [i 1 max-idx 0 max-val (.entry x 0)]
        (if (< i cnt)
          (let [v (.entry x i)]
            (if (< max-val v)
              (recur (inc i) i v)
              (recur (inc i) max-idx max-val)))
          max-idx))
      0)))

(defn vector-imin [^RealVector x]
  (let [cnt (.dim x)]
    (if (< 0 cnt)
      (loop [i 1 min-idx 0 min-val (.entry x 0)]
        (if (< i cnt)
          (let [v (.entry x i)]
            (if (< v min-val)
              (recur (inc i) i v)
              (recur (inc i) min-idx min-val)))
          min-idx))
      0)))
