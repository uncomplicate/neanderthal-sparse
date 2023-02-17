(ns uncomplicate.neanderthal.internal.cpp.mkl.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import org.bytedeco.mkl.global.mkl_rt))

(def ^:const mkl-enable-instructions
  {:avx mkl_rt/MKL_ENABLE_AVX
   :sse42 mkl_rt/MKL_ENABLE_SSE4_2
   :avx512 mkl_rt/MKL_ENABLE_AVX512
   :avx512-e1 mkl_rt/MKL_ENABLE_AVX512_E1
   :avx512-e2 mkl_rt/MKL_ENABLE_AVX512_E2
   :avx512-e3 mkl_rt/MKL_ENABLE_AVX512_E3
   :avx512-e4 mkl_rt/MKL_ENABLE_AVX512_E4
   :avx2 mkl_rt/MKL_ENABLE_AVX2
   :avx2-e1 mkl_rt/MKL_ENABLE_AVX2_E1})

(def ^:const mkl-verbose-mode
  {:timing 2
   :log 1
   :none 0})

(defn dec-verbose-mode [^long mode]
  (case mode
    2 :timing
    1 :log
    0 :none
    (dragan-says-ex "Unknown verbose mode." {:mode mode})))

(defn dec-mkl-result [^long result]
  (case result
    1 true
    0 false
    (dragan-says-ex "Unknown MKL result type." {:result result})))

(def ^:const mkl-peak-mem
  {:report mkl_rt/MKL_PEAK_MEM
   :enable mkl_rt/MKL_PEAK_MEM_ENABLE
   :disable mkl_rt/MKL_PEAK_MEM_DISABLE
   :reset mkl_rt/MKL_PEAK_MEM_RESET})
