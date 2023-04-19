;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.sparse-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.sparse :refer [csv csr]]
            [uncomplicate.neanderthal.internal.cpp.structures :refer [entries indices];;TODO
             ]
            [uncomplicate.neanderthal.internal.cpp.mkl
             [factory :refer [mkl-float mkl-double mkl-int mkl-long mkl-short mkl-byte]]])
  (:import  clojure.lang.ExceptionInfo))

;; ================= Sparse tests ===============================

(defn test-create [factory]
  (facts "Sparse constructors."
         (with-release [x0 (csv factory 0)
                        x1 (csv factory 1)
                        xr0 (csv factory 0)
                        xr1 (csv factory 1)
                        a00 (csr factory 0 0)
                        a11 (csr factory 1 1)
                        ;; ar00 (ge factory 0 0)
                        ;; ar11 (ge factory 1 1)
                        ]
           (dim x0) => 0
           (seq (entries x0)) => []
           (seq (indices x0)) => []
           (dim x1) => 1
           (seq (entries x1)) => []
           (seq (indices x1)) => []
           (dim xr0) => 0
           (dim xr1) => 1
           (csv factory -1) => (throws ExceptionInfo)
           (mrows a00) => 0
           (ncols a00) => 0
           (mrows a11) => 1
           (ncols a11) => 1
           (csv factory -1) => (throws ExceptionInfo)
           (csr factory -1 -1) => (throws ExceptionInfo)
           (csr factory -3 0) => (throws ExceptionInfo))))

(defn test-equality [factory]
  (facts "Sparse equality and hash code tests."
         (with-release [x1 (csv factory 70 [10 20 30 40])
                        y1 (csv factory 70 [10 20 30 40])
                        y2 (csv factory 70 [1 3])
                        y3 (csv factory 60 [10 20 30 40])
                        y4 (csv factory 70 [10 20 30 40] [1.0])
                        y5 (csv factory 70)
                        a1 (csr factory 1 70 [[10 20 30 40] nil])
                        a2 (csr factory 1 70 [[10 20 30 40] nil])
                        a3 (csr factory 1 70 [[1 3] nil])
                        a4 (csr factory 1 60 [[10 20 30 40] nil])
                        a5 (csr factory 2 70)]
           (.equals x1 nil) => false
           (= x1 x1) => true
           (= x1 y1) => true
           (= x1 y2) => false
           (= x1 y3) => false
           (= x1 y4) => false
           (= x1 y5) => false
           (= a1 a1) => true
           (= a1 a2) => true
           (= a1 a3) => false
           (= a1 a4) => false
           (csr factory 2 70 [[1 3] nil]) => (throws ExceptionInfo))))

(defn test-release [factory]
  (let [x (csv factory 3 [1] [1.0])
        a (csr factory 2 3 [[1] [1.0]
                            [1] [2.0]])
        ;;col-a (col a 0)
        ]
    (facts "CSVector and CSRMatrix release tests."
           ;; (release col-a) => true
           ;; (release col-a) => true
           ;; (release sub-a) => true
           ;; (release sub-a) => true
           (release x) => true
           (release x) => true
           (release a) => true
           (release a) => true
           )))

(defn test-csv-transfer [factory0 factory1]
  (with-release [x0 (csv factory0 4 [1 3])
                 x1 (csv factory0 4 [1 3] [1.0 2.0])
                 x2 (csv factory0 4 [1 3] [10 20])
                 y0 (csv factory1 55 [22 33])
                 y1 (csv factory1 55 [22 33] [10 20])
                 y2 (csv factory1 55 [22 33] [10 20])]
    (facts
     "Vector transfer tests."
     (transfer! (float-array [1 2 3]) x0) => x1
     (transfer! (double-array [1 2 3]) x0) => x1
     (transfer! (int-array [1 2 3 0 44]) x0) => x1
     (transfer! (long-array [1 2 3]) x0) => x1
     (seq (transfer! x1 (float-array 2))) => [1.0 2.0]
     (seq (transfer! x1 (double-array 2))) => [1.0 2.0]
     (transfer! y1 x0) => x2
     (transfer! x2 y0) => y2)))

(test-create mkl-float)
(test-equality mkl-float)
(test-release mkl-float)
(test-csv-transfer mkl-float mkl-float)
