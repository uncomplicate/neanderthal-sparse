;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://openpsource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.sparse-blas
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release extract]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [position!]]
            [uncomplicate.neanderthal.block :refer [buffer offset]]
            [uncomplicate.neanderthal.internal.cpp.mkl.constants :refer :all])
  (:import [org.bytedeco.javacpp Pointer]
           [org.bytedeco.mkl.global mkl_rt]))
