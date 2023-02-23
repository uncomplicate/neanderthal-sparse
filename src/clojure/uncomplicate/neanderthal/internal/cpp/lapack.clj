;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.lapack
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [block :refer [offset stride]]])
  (:import [org.bytedeco.mkl.global mkl_rt]))

(defmacro with-lapack-check [expr]
  ` (let [err# ~expr]
      (if (zero? err#)
        err#
        (throw (ex-info "LAPACK error." {:error-code err# :bad-argument (- err#)})))))

;; TODO remove
(defmacro vctr-laset [method ptr alpha x]
  `(do
     (with-lapack-check
       (~method mkl_rt/CblasRowMajor (byte (int \g)) (dim ~x) 1 ~alpha ~alpha (~ptr ~x) (stride ~x)))
     ~x))

(defmacro vctr-lasrt [method ptr x increasing]
  `(if (= 1 (stride ~x))
     (do
       (with-lapack-check
         (~method (byte (int (if ~increasing \I \D))) (dim ~x) (~ptr ~x)))
       ~x)
     (dragan-says-ex "You cannot sort a vector with stride different than 1." {:stride (stride ~x)})))
