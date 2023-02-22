;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.sparse
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release]]
             [utils :refer [dragan-says-ex direct-buffer]]]
            [uncomplicate.clojure-cpp :refer :all]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! subvector vctr ge]]
             [real :refer [entry entry!]]
             [math :refer [ceil]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer :all]])
  (:import [org.bytedeco.mkl.global mkl_rt]))

(declare integer-sparse-vector real-sparse-vector)


;; (use 'criterium.core)

;; (def m 2)
;; (def p 2)
;; (def n 1)

;; (def a ^org.bytedeco.javacpp.DoublePointer (.capacity (double-pointer (mkl_rt/MKL_malloc (* m p 8) 64)) (* m p))) ;; NOTE: deallocator is null
;; (def b ^org.bytedeco.javacpp.DoublePointer (.capacity (double-pointer (mkl_rt/MKL_malloc (* p n 8) 64)) (* m p)))
;; (def c ^org.bytedeco.javacpp.DoublePointer (.capacity (double-pointer (mkl_rt/MKL_malloc (* m n 8) 64)) (* m p)))

;; (mkl_rt/cblas_dgemm mkl_rt/CblasRowMajor, mkl_rt/CblasNoTrans mkl_rt/CblasNoTrans ^int m ^int n ^int p 1.0 ^org.bytedeco.javacpp.DoublePointer a ^int p ^org.bytedeco.javacpp.DoublePointer b ^int n 0.0 ^org.bytedeco.javacpp.DoublePointer c ^int n)
