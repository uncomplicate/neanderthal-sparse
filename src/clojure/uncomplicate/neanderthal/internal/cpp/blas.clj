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
            [uncomplicate.neanderthal.internal
             [api :refer [iamax engine navigator storage region]]
             [common :refer [check-eq-navigators]]
             [navigation :refer [full-storage]]]
            [uncomplicate.neanderthal.internal.cpp.mkl.constants :refer :all])
  (:import [org.bytedeco.mkl.global mkl_rt]
           [org.bytedeco.javacpp Pointer FloatPointer DoublePointer IntPointer]
           [uncomplicate.neanderthal.internal.api VectorSpace Block RealVector]))

;; TODO move to structures

(defn float-ptr
  (^FloatPointer [^Block x]
   (.buffer x))
  (^FloatPointer [^Block x ^long i]
   (.position (FloatPointer. ^FloatPointer (.buffer x)) i)))

(defn double-ptr
  (^DoublePointer [^Block x]
   (.buffer x))
  (^DoublePointer [^Block x ^long i]
   (.position (DoublePointer. ^DoublePointer (.buffer x)) i)))

(defn int-ptr ^IntPointer [^Block x]
  (.buffer x))

(defn coerce-float-ptr
  (^FloatPointer [^Block x]
   (FloatPointer. ^Pointer (.buffer x)))
  (^FloatPointer [^Block x ^long i]
   (.position (FloatPointer. ^Pointer (.buffer x)) i)))

(defn coerce-double-ptr ^DoublePointer
  (^DoublePointer [^Block x]
   (DoublePointer. ^Pointer (.buffer x)))
  (^DoublePointer [^Block x ^long i]
   (.position (DoublePointer. ^Pointer (.buffer x)) i)))

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

(defmacro full-storage-reduce
  ([a b len buff-a buff-b ld-b acc init expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~a)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          n# (.fd stor-a#)]
      (if (= nav-a# nav-b#)
        (if (and (.isGapless stor-a#) (.isGapless stor-b#))
          ~expr-direct
          (let [~ld-b 1]
            (loop [j# 0 ~acc ~init]
              (if (< j# n#)
                (recur (inc j#)
                       (let [start# (.start nav-a# reg# j#)
                             ~len (- (.end nav-a# reg# j#) start#)]
                         (position! ~buff-a (.index stor-a# start# j#))
                         (position! ~buff-b (.index stor-b# start# j#))
                         ~expr))
                ~acc))))
        (let [~ld-b (.ld stor-b#)]
          (loop [j# 0 ~acc ~init]
            (if (< j# n#)
              (recur (inc j#)
                     (let [start# (.start nav-a# reg# j#)
                           ~len (- (.end nav-a# reg# j#) start#)]
                       (position! ~buff-a (.index stor-a# start# j#))
                       (position! ~buff-b (.index stor-b# j# start#))
                       ~expr))
              ~acc)))))))

(defmacro full-storage-map
  ([a b len buff-a buff-b ld-a expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-b# (.fd stor-b#)]
      (if (= nav-a# nav-b#)
        (if (and (.isGapless stor-a#) (.isGapless stor-b#))
          ~expr-direct
          (let [~ld-a 1]
            (dotimes [j# fd-b#]
              (let [start# (.start nav-b# reg# j#)
                    ~len (- (.end nav-b# reg# j#) start#)]
                (position! ~buff-a (.index stor-a# start# j#))
                (position! ~buff-b (.index stor-b# start# j#))
                ~expr))))
        (let [~ld-a (.ld stor-a#)]
          (dotimes [j# fd-b#]
            (let [start# (.start nav-b# reg# j#)
                  ~len (- (.end nav-b# reg# j#) start#)]
              (position! ~buff-a (.index stor-a# j# start#))
              (position! ~buff-b (.index stor-b# start# j#))
              ~expr)))))))

(defmacro full-matching-map
  ([a b len buff-a buff-b expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-a# (.fd stor-a#)]
      (check-eq-navigators ~a ~b)
      (if (and (.isGapless stor-a#) (.isGapless stor-b#))
        ~expr-direct
        (dotimes [j# fd-a#]
          (let [start# (.start nav-a# reg# j#)
                ~len (- (.end nav-a# reg# j#) start#)]
            (position! ~buff-a (.index stor-a# start# j#))
            (position! ~buff-b (.index stor-b# start# j#))
            ~expr)))))
  ([a b c len buff-a buff-b buff-c expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          nav-c# (navigator ~c)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          stor-c# (full-storage ~c)
          fd-a# (.fd stor-a#)]
      (check-eq-navigators ~a ~b ~c)
      (if (and (.isGapless stor-a#) (.isGapless stor-b#) (.isGapless stor-c#))
        ~expr-direct
        (dotimes [j# fd-a#]
          (let [start# (.start nav-a# reg# j#)
                ~len (- (.end nav-a# reg# j#) start#)]
            (position! ~buff-a (.index stor-a# start# j#))
            (position! ~buff-b (.index stor-b# start# j#))
            (position! ~buff-c (.index stor-c# start# j#))
            ~expr))))))

(defmacro ge-map [blas method ptr a b]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)]
         (full-storage-map ~a ~b len# buff-a# buff-b# ld-a#
                           (. ~blas ~method (dim ~a) buff-a# 1 buff-b# 1)
                           (. ~blas ~method len# buff-a# ld-a# buff-b# 1))))
     ~a))

(defmacro ge-reduce [blas method ptr init a b]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)]
       (full-storage-reduce ~a ~b len# buff-a# buff-b# ld-b# acc# ~init
                            (. ~blas ~method (dim ~a) buff-a# 1 buff-b# 1)
                            (+ acc# (. ~blas ~method len# buff-a# 1 buff-b# ld-b#))))
     ~init))
