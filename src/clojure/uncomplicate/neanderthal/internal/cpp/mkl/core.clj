;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.core
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release]]
             [utils :refer [dragan-says-ex direct-buffer]]]
            [uncomplicate.clojure-cpp :refer [pointer-seq]]
            [uncomplicate.neanderthal.internal.cpp.mkl.constants :refer :all])
  (:import [org.bytedeco.javacpp Pointer]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$MKLVersion]))

;; ===================== System ================================================================

(defn version []
  (with-release [result (mkl_rt$MKLVersion.)]
    (mkl_rt/MKL_Get_Version result)
    {:major (.MajorVersion result)
     :minor (.MinorVersion result)
     :update (.UpdateVersion result)}))

;; ===================== Memory Management =====================================================

(defn malloc!
  ([^long size]
   (mkl_rt/MKL_malloc size -1))
  ([^long size ^long align]
   (mkl_rt/MKL_malloc size align)))

(defn calloc!
  ([^long n ^long element-size]
   (mkl_rt/MKL_calloc n element-size -1))
  ([^long n ^long element-size ^long align]
   (mkl_rt/MKL_calloc n element-size align)))

(defn realloc! [^Pointer p ^long size]
  (mkl_rt/MKL_realloc p size))

(defn free! [^Pointer p]
  (mkl_rt/MKL_free p)
  (.setNull p)
  p)

(defn free-buffers! []
  (mkl_rt/MKL_Free_Buffers))

(defn thread-free-buffers! []
  (mkl_rt/MKL_Thread_Free_Buffers))

(defn allocated-bytes ^long []
  (mkl_rt/MKL_Mem_Stat (int-array 3)))

(defn peak-mem-usage! ^long [mode]
  (let [result (mkl_rt/MKL_Peak_Mem_Usage (get mkl-peak-mem mode mode))]
    (if (= -1 result)
      (dragan-says-ex "MKL reported an unspecified error during this call.")
      result)))

(defn peak-mem-usage ^long []
  (peak-mem-usage! :report))

;; ===================== Miscellaneous =========================================================

(defn enable-instructions! [instruction-set]
  (dec-mkl-result (mkl_rt/MKL_Enable_Instructions (get mkl-enable-instructions instruction-set instruction-set))))

(defn verbose! [timing]
  (dec-verbose-mode (mkl_rt/MKL_Verbose (get mkl-verbose-mode timing timing))))
