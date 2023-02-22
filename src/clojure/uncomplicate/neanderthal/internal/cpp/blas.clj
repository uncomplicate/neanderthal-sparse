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
            [uncomplicate.clojure-cpp :refer [position! pointer]]
            [uncomplicate.neanderthal
             [block :refer [buffer offset stride]]
             [core :refer [dim entry]]]
            [uncomplicate.neanderthal.internal.api :refer [iamax engine data-accessor]]
            [uncomplicate.neanderthal.internal.cpp.mkl.constants :refer :all])
  (:import [org.bytedeco.mkl.global mkl_rt]
           [org.bytedeco.javacpp FloatPointer DoublePointer]
           [uncomplicate.neanderthal.internal.api VectorSpace Block RealVector]))

#_(defn cpp-pointer [constructor x]
  (position! (constructor (buffer x)) (offset x)))

(defn float-ptr ^FloatPointer [^Block x]
  (.buffer x))

(defn double-ptr ^DoublePointer [^Block x]
  (.buffer x))

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
