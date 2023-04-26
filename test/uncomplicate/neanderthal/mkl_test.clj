;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.mkl-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer [ge]]
             [native :refer [factory-by-type]]
             [block-test :as block-test]
             [real-test :as real-test]
             [math-test :as math-test]
             [random-test :as random-test]]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory
             :refer [mkl-float mkl-double mkl-int mkl-long mkl-short mkl-byte]]))

(defn test-all [factory]
  (block-test/test-create factory)
  (block-test/test-equality factory)
  (block-test/test-release factory)
  (block-test/test-vctr-op factory)
  (block-test/test-ge-op factory)
  (block-test/test-vctr-contiguous factory)
  (block-test/test-ge-contiguous factory))

(defn test-two-factories [factory0 factory1]
  (block-test/test-vctr-transfer factory0 factory1)
  (block-test/test-ge-transfer factory0 factory1))

(defn test-host [factory]
  (block-test/test-vctr-ifn factory)
  (block-test/test-vctr-functor factory)
  (block-test/test-vctr-fold factory)
  (block-test/test-vctr-reducible factory)
  (block-test/test-vctr-seq factory)
  (block-test/test-vctr-functor-laws factory)
  (block-test/test-ge-functor-laws factory ge))

(test-all mkl-double)
(test-all mkl-float)

(for [factory0 [mkl-double mkl-float]
      factory1 [mkl-long mkl-int mkl-short mkl-byte]]
  (test-two-factories factory0 factory1))

(for [factory0 [mkl-double mkl-float]
      factory1 [mkl-double mkl-float]]
  [(test-two-factories factory0 factory1)
   (test-two-factories factory1 factory0)])

(test-host mkl-float)
(test-host mkl-double)

;; ================= Sparse tests ===============================
