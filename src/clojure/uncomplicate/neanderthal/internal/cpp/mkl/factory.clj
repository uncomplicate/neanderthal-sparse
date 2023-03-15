;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.factory
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release]]
             [utils :refer [dragan-says-ex generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojure-cpp :refer [long-pointer float-pointer double-pointer put!]]
            [uncomplicate.neanderthal
             [core :refer [dim entry mrows ncols] :as core]
             [math :refer [f=]]
             [block :refer [create-data-source initialize buffer offset stride]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage accu-layout dostripe-layout]]
             [common :refer [check-stride check-eq-navigators]]]
            [uncomplicate.neanderthal.internal.cpp
             [structures :refer :all]
             [lapack :refer :all]
             [blas :refer [float-ptr double-ptr int-ptr coerce-double-ptr coerce-float-ptr
                           vector-imax vector-imin ge-map ge-reduce]]]
            [uncomplicate.neanderthal.internal.cpp.mkl.core :refer [malloc!]]
            [uncomplicate.neanderthal.internal.host.mkl
             :refer [sigmoid-over-tanh vector-ramp vector-relu vector-elu]])
  (:import [uncomplicate.neanderthal.internal.api DataAccessor Block Vector LayoutNavigator Region
            GEMatrix DenseStorage]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$VSLStreamStatePtr]))

;; =============== Factories ==================================================

(def ^:const blas-layout
  {:row mkl_rt/CblasRowMajor
   :column mkl_rt/CblasColMajor})

(def ^:const blas-transpose
  {:trans mkl_rt/CblasTrans
   :no-trans mkl_rt/CblasNoTrans})

(declare mkl-int)

(defn cblas
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (cblas "cblas_" type name)))

(defn lapacke
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (lapacke "LAPACKE_" type name)))

(defn math
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (math "v" type name)))

;; ================= Integer Vector Engines =====================================

(def ^{:no-doc true :const true} INTEGER_UNSUPPORTED_MSG
  "Integer BLAS operations are not supported. Please transform data to float or double.")

(def ^{:no-doc true :const true} SHORT_UNSUPPORTED_MSG
  "BLAS operation on short vectors are supported only on dimensions divisible by 2 (short) or 4 (byte).")

(defmacro integer-vector-blas* [name t ptr blas]
  `(extend-type ~name
     Blas
     (swap [_# x# y#]
       (. ~blas ~(cblas t 'swap) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       x#)
     (copy [_# x# y#]
       (. ~blas ~(cblas t 'copy) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     (dot [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrm1 [_# x#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrm2 [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrmi [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (asum [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (iamax [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (iamin [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rot [_# _# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rotg [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rotm [_# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rotmg [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (scal [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (axpy [_# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     Lapack
     (srt [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))))

(defmacro integer-vector-blas-plus* [name t cast ptr blas lapack]
  `(extend-type ~name
     BlasPlus
     (amax [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (subcopy [_# x# y# kx# lx# ky#]
       (. ~blas ~(cblas t 'copy) (int lx#) (~ptr x# kx#) (stride x#) (~ptr y# ky#) (stride y#))
       y#)
     (sum [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (imax [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (imin [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (set-all [_# alpha# x#]
       (with-lapack-check
         (. ~lapack ~(lapacke t 'laset) ~(int (:row blas-layout)) ~(byte (int \g)) (dim x#) 1
            (~cast alpha#) (~cast alpha#) (~ptr x#) (stride x#)))
       x#)
     (axpby [_# _# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))))

;; ================= Real Vector Engines ========================================

(defmacro real-vector-blas* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     Blas
     (swap [this# x# y#]
       (. ~blas ~(cblas t 'swap) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       x#)
     (copy [this# x# y#]
       (. ~blas ~(cblas t 'copy) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     (dot [this# x# y#]
       (. ~blas ~(cblas t 'dot) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#)))
     (nrm1 [this# x#]
       (asum this# x#))
     (nrm2 [this# x#]
       (. ~blas ~(cblas t 'nrm2) (dim x#) (~ptr x#) (stride x#)))
     (nrmi [this# x#]
       (amax this# x#))
     (asum [this# x#]
       (. ~blas ~(cblas t 'asum) (dim x#) (~ptr x#) (stride x#)))
     (iamax [this# x#]
       (. ~blas ~(cblas 'cblas_i t 'amax) (dim x#) (~ptr x#) (stride x#)))
     (iamin [this# x#]
       (. ~blas ~(cblas 'cblas_i t 'amin) (dim x#) (~ptr x#) (stride x#)))
     (rot [this# x# y# c# s#]
       (. ~blas ~(cblas t 'rot) (dim x#)
          (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~cast c#) (~cast s#)))
     (rotg [this# abcs#]
       (check-stride abcs#)
       (. ~blas ~(cblas t 'rotg) (~ptr abcs#) (~ptr abcs# 1) (~ptr abcs# 2) (~ptr abcs# 3))
       abcs#)
     (rotm [this# x# y# param#]
       (. ~blas ~(cblas t 'rotm) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr param#)))
     (rotmg [this# d1d2xy# param#]
       (check-stride d1d2xy# param#)
       (. ~blas ~(cblas t 'rotmg)
          (~ptr d1d2xy#) (~ptr d1d2xy# 1) (~ptr d1d2xy# 2) (~cast (entry d1d2xy# 3)) (~ptr param#)))
     (scal [this# alpha# x#]
       (. ~blas ~(cblas t 'scal) (dim x#) (~cast alpha#) (~ptr x#) (stride x#))
       x#)
     (axpy [this# alpha# x# y#]
       (. ~blas ~(cblas t 'axpy) (dim x#) (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     BlasPlus
     (amax [this# x#]
       (if (< 0 (dim x#))
         (Math/abs (double (entry x# (iamax this# x#))))
         0.0))
     (subcopy [this# x# y# kx# lx# ky#]
       (. ~blas ~(cblas t 'copy) (int lx#) (~ptr x# kx#) (stride x#) (~ptr y# ky#) (stride y#))
       y#)
     (sum [this# x#]
       (. ~blas ~(cblas t 'dot) (dim x#) (~ptr x#) (stride x#) (~ptr ~ones) 0))
     (imax [this# x#]
       (vector-imax x#))
     (imin [this# x#]
       (vector-imin x#))
     (set-all [this# alpha# x#]
       (with-lapack-check
         (. ~lapack ~(lapacke t 'laset) ~(int (:row blas-layout)) ~(byte (int \g)) (dim x#) 1
            (~cast alpha#) (~cast alpha#) (~ptr x#) (stride x#)))
       x#)
     (axpby [this# alpha# x# beta# y#]
       (. ~blas ~(cblas t 'axpby) (dim x#)
          (~cast alpha#) (~ptr x#) (stride x#) (~cast beta#) (~ptr y#) (stride y#))
       y#)
     Lapack
     (srt [this# x# increasing#]
       (if (= 1 (stride x#))
         (with-lapack-check
           (. ~lapack ~(lapacke t 'lasrt) (byte (int (if increasing# \I \D))) (dim x#) (~ptr x#)))
         (dragan-says-ex "You cannot sort a vector with stride." {"stride" (stride x#)}))
       x#)))

(defmacro vector-math
  ([method ptr a y]
   `(do
      (check-stride ~a ~y)
      (. mkl_rt ~method (dim ~a) (~ptr ~a) (~ptr ~y))
      ~y))
  ([method ptr a b y]
   `(do
      (check-stride ~a ~b ~y)
      (. mkl_rt ~method (dim ~a) (~ptr ~a) (~ptr ~b) (~ptr ~y))
      ~y)))

(defmacro with-rng-check [x expr]
  `(if (< 0 (dim ~x))
     (if (= 1 (stride ~x))
       (let [err# ~expr]
         (if (= 0 err#)
           ~x
           (throw (ex-info "MKL error." {:error-code err#}))))
       (dragan-says-ex "This engine cannot generate random entries in host vectors with stride. Sorry."
                       {:v (info ~x)}))
     ~x))

(defmacro with-mkl-check [expr res]
  `(let [err# ~expr]
     (if (zero? err#)
       ~res
       (throw (ex-info "MKL error." {:error-code err#})))))

(defn create-stream-ars5 ^mkl_rt$VSLStreamStatePtr [seed]
  (let-release [stream (mkl_rt$VSLStreamStatePtr. (long-pointer 1))]
    (with-mkl-check (mkl_rt/vslNewStream stream mkl_rt/VSL_BRNG_ARS5 seed)
      stream)))

(def ^:private default-rng-stream (create-stream-ars5 (generate-seed)))

(defmacro real-vector-math* [name t ptr cast]
  `(extend-type ~name
     VectorMath
     (sqr [_# a# y#]
       (vector-math ~(math t 'Sqr) ~ptr a# y#))
     (mul [_# a# b# y#]
       (vector-math ~(math t 'Mul) ~ptr a# b# y#))
     (div [_# a# b# y#]
       (vector-math ~(math t 'Div) ~ptr a# b# y#))
     (inv [_# a# y#]
       (vector-math ~(math t 'Inv) ~ptr a# y#))
     (abs [_# a# y#]
       (vector-math ~(math t 'Abs) ~ptr a# y#))
     (linear-frac [_# a# b# scalea# shifta# scaleb# shiftb# y#]
       (check-stride a# b# y#)
       (. mkl_rt ~(math t 'LinearFrac) (dim a#) (~ptr a#) (~ptr b#)
          (~cast scalea#) (~cast shifta#) (~cast scaleb#) (~cast shiftb#) (~ptr y#))
       y#)
     (fmod [_# a# b# y#]
       (vector-math ~(math t 'Fmod) ~ptr a# b# y#))
     (frem [_# a# b# y#]
       (vector-math  ~(math t 'Remainder) ~ptr a# b# y#))
     (sqrt [_# a# y#]
       (vector-math ~(math t 'Sqrt) ~ptr a# y#))
     (inv-sqrt [_# a# y#]
       (vector-math ~(math t 'InvSqrt) ~ptr a# y#))
     (cbrt [_# a# y#]
       (vector-math ~(math t 'Cbrt) ~ptr a# y#))
     (inv-cbrt [_# a# y#]
       (vector-math ~(math t 'InvCbrt) ~ptr a# y#))
     (pow2o3 [_# a# y#]
       (vector-math ~(math t 'Pow2o3) ~ptr a# y#))
     (pow3o2 [_# a# y#]
       (vector-math ~(math t 'Pow3o2) ~ptr a# y#))
     (pow [_# a# b# y#]
       (vector-math ~(math t 'Pow) ~ptr a# b# y#))
     (powx [_# a# b# y#]
       (check-stride a# y#)
       (. mkl_rt ~(math t 'Powx) (dim a#) (~ptr a#) (~cast b#) (~ptr y#))
       y#)
     (hypot [_# a# b# y#]
       (vector-math ~(math t 'Hypot) ~ptr a# b# y#))
     (exp [_# a# y#]
       (vector-math ~(math t 'Exp) ~ptr a# y#))
     (exp2 [_# a# y#]
       (vector-math ~(math t 'Exp2) ~ptr a# y#))
     (exp10 [_# a# y#]
       (vector-math ~(math t 'Exp10) ~ptr a# y#))
     (expm1 [_# a# y#]
       (vector-math ~(math t 'Expm1) ~ptr a# y#))
     (log [_# a# y#]
       (vector-math ~(math t 'Ln) ~ptr a# y#))
     (log2 [_# a# y#]
       (vector-math ~(math t 'Log2) ~ptr a# y#))
     (log10 [_# a# y#]
       (vector-math ~(math t 'Log10) ~ptr a# y#))
     (log1p [_# a# y#]
       (vector-math ~(math t 'Log1p) ~ptr a# y#))
     (sin [_# a# y#]
       (vector-math ~(math t 'Sin) ~ptr a# y#))
     (cos [_# a# y#]
       (vector-math ~(math t 'Cos) ~ptr a# y#))
     (tan [_# a# y#]
       (vector-math ~(math t 'Tan) ~ptr a# y#))
     (sincos [_# a# y# z#]
       (vector-math ~(math t 'SinCos) ~ptr a# y# z#))
     (asin [_# a# y#]
       (vector-math ~(math t 'Asin) ~ptr a# y#))
     (acos [_# a# y#]
       (vector-math ~(math t 'Acos) ~ptr a# y#))
     (atan [_# a# y#]
       (vector-math ~(math t 'Atan) ~ptr a# y#))
     (atan2 [_# a# b# y#]
       (vector-math ~(math t 'Atan2) ~ptr a# b# y#))
     (sinh [_# a# y#]
       (vector-math ~(math t 'Sinh) ~ptr a# y#))
     (cosh [_# a# y#]
       (vector-math ~(math t 'Cosh) ~ptr a# y#))
     (tanh [_# a# y#]
       (vector-math ~(math t 'Tanh) ~ptr a# y#))
     (asinh [_# a# y#]
       (vector-math ~(math t 'Asinh) ~ptr a# y#))
     (acosh [_# a# y#]
       (vector-math ~(math t 'Acosh) ~ptr a# y#))
     (atanh [_# a# y#]
       (vector-math ~(math t 'Atanh) ~ptr a# y#))
     (erf [_# a# y#]
       (vector-math ~(math t 'Erf) ~ptr a# y#))
     (erfc [_# a# y#]
       (vector-math ~(math t 'Erfc) ~ptr a# y#))
     (erf-inv [_# a# y#]
       (vector-math ~(math t 'ErfInv) ~ptr a# y#))
     (erfc-inv [_# a# y#]
       (vector-math ~(math t 'ErfcInv) ~ptr a# y#))
     (cdf-norm [_# a# y#]
       (vector-math ~(math t 'CdfNorm) ~ptr a# y#))
     (cdf-norm-inv [_# a# y#]
       (vector-math ~(math t 'CdfNormInv) ~ptr a# y#))
     (gamma [_# a# y#]
       (vector-math ~(math t 'TGamma) ~ptr a# y#))
     (lgamma [_# a# y#]
       (vector-math ~(math t 'LGamma) ~ptr a# y#))
     (expint1 [_# a# y#]
       (vector-math ~(math t 'ExpInt1) ~ptr a# y#))
     (floor [_# a# y#]
       (vector-math ~(math t 'Floor) ~ptr a# y#))
     (fceil [_# a# y#]
       (vector-math ~(math t 'Ceil) ~ptr a# y#))
     (trunc [_# a# y#]
       (vector-math ~(math t 'Trunc) ~ptr a# y#))
     (round [_# a# y#]
       (vector-math ~(math t 'Round) ~ptr a# y#))
     (modf [_# a# y# z#]
       (vector-math ~(math t 'Modf) ~ptr a# y# z#))
     (frac [_# a# y#]
       (vector-math ~(math t 'Frac) ~ptr a# y#))
     (fmin [_# a# b# y#]
       (vector-math ~(math t 'Fmin) ~ptr a# b# y#))
     (fmax [_# a# b# y#]
       (vector-math ~(math t 'Fmax) ~ptr a# b# y#))
     (copy-sign [_# a# b# y#]
       (vector-math ~(math t 'CopySign) ~ptr a# b# y#))
     (sigmoid [this# a# y#]
       (sigmoid-over-tanh this# a# y#))
     (ramp [this# a# y#]
       (vector-ramp this# a# y#))
     (relu [this# alpha# a# y#]
       (vector-relu this# alpha# a# y#))
     (elu [this# alpha# a# y#]
       (vector-elu this# alpha# a# y#))))

(defn float-uniform [^mkl_rt$VSLStreamStatePtr stream ^double lower ^double upper x]
  (with-rng-check x
    (mkl_rt/vsRngUniform mkl_rt/VSL_RNG_METHOD_UNIFORM_STD stream (dim x) (float-ptr x) lower upper)))

(defn float-gaussian [^mkl_rt$VSLStreamStatePtr stream ^double mu ^double sigma x]
  (with-rng-check x
    (mkl_rt/vsRngGaussian mkl_rt/VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 stream (dim x) (float-ptr x) mu sigma)))

(defn double-uniform [^mkl_rt$VSLStreamStatePtr stream ^double lower ^double upper x]
  (with-rng-check x
    (mkl_rt/vdRngUniform mkl_rt/VSL_RNG_METHOD_UNIFORM_STD stream (dim x) (double-ptr x) lower upper)))

(defn double-gaussian [^mkl_rt$VSLStreamStatePtr stream ^double mu ^double sigma x]
  (with-rng-check x
    (mkl_rt/vdRngGaussian mkl_rt/VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 stream (dim x) (double-ptr x) mu sigma)))

(def ^:private ones-float (->RealBlockVector nil nil nil true
                                             (doto (float-pointer 1) (put! 0 1.0)) 1 0))
(def ^:private ones-double (->RealBlockVector nil nil nil true
                                              (doto (double-pointer 1) (put! 0 1.0)) 1 0))

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-vector-math* FloatVectorEngine "s" float-ptr float)

(extend-type FloatVectorEngine
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (float-uniform (or rng-stream default-rng-stream) lower upper x))
  (rand-normal [this rng-stream mu sigma x]
    (with-rng-check x
      (float-gaussian (or rng-stream default-rng-stream) mu sigma x))))

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-vector-math* DoubleVectorEngine "d" double-ptr double)

(extend-type DoubleVectorEngine
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (double-uniform (or rng-stream default-rng-stream) lower upper x))
  (rand-normal [this rng-stream mu sigma x]
    (with-rng-check x
      (double-gaussian (or rng-stream default-rng-stream) mu sigma x))))

(deftype LongVectorEngine [])
(integer-vector-blas* LongVectorEngine "d" coerce-double-ptr mkl_rt)
(integer-vector-blas-plus* LongVectorEngine "d" double coerce-double-ptr mkl_rt mkl_rt)

(deftype IntVectorEngine [])
(integer-vector-blas* IntVectorEngine "s" coerce-float-ptr mkl_rt)
(integer-vector-blas-plus* IntVectorEngine "s" float coerce-float-ptr mkl_rt mkl_rt)

;; ================= Real GE Engine ========================================

(defmacro ge-axpby [blas method ptr alpha a beta b]
  `(if (< 0 (dim ~a))
     (let [nav-b# (navigator ~b)]
       (. ~blas ~method (byte (int (if (.isColumnMajor nav-b#) \C \R)))
          (byte (int (if (= (navigator ~a) nav-b#) \n \t))) ~(byte (int \n)) (mrows ~b) (ncols ~b)
          ~alpha (~ptr ~a) (stride ~a) ~beta (~ptr ~b) (stride ~b)
          (~ptr ~b) (stride ~b))
       ~b)
     ~b))

(defmacro real-ge-blas* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (ge-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (when (< 0 (dim a#))
         (let [stor-b# (full-storage b#)
               no-trans# (= (navigator a#) (navigator b#))]
           (. ~blas ~(cblas "mkl_" t 'omatcopy) ~(byte (int \C)) (byte (int (if no-trans# \N \T)))
              (if no-trans# (.sd stor-b#) (.fd stor-b#)) (if no-trans# (.fd stor-b#) (.sd stor-b#))
              1.0 (~ptr a#) (stride a#) (~ptr b#) (.ld stor-b#))))
       b#)
     (dot [_# a# b#]
       (ge-reduce ~blas ~(cblas t 'dot) ~ptr 0.0 a# b#))
     (nrm1 [_# x#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr ~(byte (int \O)) a))
     (nrm2 [_# x#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr ~(byte (int \F)) a))
     (nrmi [_# x#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr ~(byte (int \I)) a))
     (asum [_# a#]
       (if (< 0 (dim a#))
         (let [buff# (~ptr a# 0)]
           (if (.isGapless (storage a#))
             (. ~blas ~(cblas t 'asum) (dim a#) (~ptr a#) 1)
             (accu-layout a# len# idx# acc# 0.0
                          (+ acc# (. ~blas ~(cblas t 'asum) len# (.position buff# idx#) 1)))))
         0.0))
     (scal [_# alpha# a#]
       (when (< 0 (dim a#))
         (let [stor# (full-storage a#)]
           (. ~blas ~(cblas "mkl_" t 'imatcopy) ~(byte (int \c)) ~(byte (int \n))
              (.sd stor#) (.fd stor#) (~cast alpha#) (~ptr a#) (.ld stor#) (.ld stor#))))
       a#)
     (axpy [_# alpha# a# b#]
       (ge-axpby ~blas ~(cblas "mkl_" t 'omatadd) ~ptr (~cast alpha#) a# 1.0 b#)
       b#)
     (mv
       ([_# alpha# a# x# beta# y#]
        (. ~blas ~(cblas t 'gemv) (.layout (navigator a#)) ~(:no-trans blas-transpose) (mrows a#) (ncols a#)
           (~cast alpha#) (~ptr a#) (stride a#) (~ptr x#) (stride x#) (~cast beta#) (~ptr y#) (stride y#))
        y#)
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported for GE matrices." {:a (info a#)})))
     (rk [_# alpha# x# y# a#]
       (. ~blas ~(cblas t 'ger) (.layout (navigator a#)) (mrows a) (ncols a#)
          (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr a#) (stride a#))
       a#)
     (mm
       ([_# alpha# a# b# _#]
        (if-not (instance? GEMatrix b#)
          (mm (engine b#) alpha# b# a# false)
          (dragan-says-ex "In-place mm! is not supported for GE matrices. Use QR factorization."
                          {:a (info a#) :b (info b#)} )))
       ([_# alpha# a# b# beta# c# _#]
        (if (instance? GEMatrix b#)
          (let [nav# (navigator c#)]
            (. ~blas ~(cblas t 'gemm) (.layout nav#)
               (if (= nav# (navigator a#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
               (if (= nav# (navigator b#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
               (mrows a#) (ncols b#) (ncols a#) (~cast alpha#) (~ptr a#) (stride a#)
               (~ptr b#) (stride b#) (~cast beta#) (~ptr c#) (stride c#))
            c#)
          (mm (engine b#) (~cast alpha#) b# a# (~cast beta#) c# false))))))

(defmacro real-ge-blas-plus* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr ~(byte (int \M)) a))
     (sum [_# a#]
       (if (< 0 (dim a#))
         (if (.isGapless (storage a#))
           (. ~blas ~(cblas t 'dot) (dim a#) (~ptr a#) 1 (~ptr ~ones) 0)
           (let [buff# (~ptr a# 0)
                 ones# (~ptr ~ones)]
             (accu-layout a# len# idx# acc# 0.0
                          (+ acc# (double (. ~blas ~(cblas t 'dot) len# (.position buff# idx#) 1 ones# 0))))))
         0.0))
     (set-all [_# alpha# a#]
       (with-lapack-check
         (. ~lapack ~(lapacke t 'laset) (.layout (navigator a#)) ~(byte (int \g))
          (mrows a#) (ncols a#) (~cast alpha#) (~cast alpha#) (~ptr a#) (stride a#)))
       a#)
     (axpby [_# alpha# a# beta# b#]
       (ge-axpby ~blas ~(cblas "mkl_" t 'omatadd) ~ptr (~cast alpha#) a# (~cast beta#) b#)
       b#)
     Lapack
     (srt [_# a# increasing#]
       (let [incr# (byte (int (if increasing# \I \D)))
             buff# (~ptr a# 0)]
         (dostripe-layout a# len# idx#
                          (with-lapack-check
                            (. ~lapack ~(lapacke t 'lasrt) incr# len# (.position buff# idx#)))))
       a#)))

(defmacro real-ge-lapack* [name t ptr cast lapack]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (let [incr# (byte (int (if increasing# \I \D)))
             buff# (~ptr a# 0)]
         (dostripe-layout a# len# idx#
                          (with-lapack-check
                            (. ~lapack ~(lapacke t 'lasrt) incr# len# (.position buff# idx#)))))
       a#)))

(deftype FloatGEEngine [])
(real-ge-blas* FloatGEEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-ge-blas-plus* FloatGEEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-ge-lapack* FloatGEEngine "s" float-ptr float mkl_rt)

(deftype DoubleGEEngine [])
(real-ge-blas* DoubleGEEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-ge-blas-plus* DoubleGEEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-ge-lapack* DoubleGEEngine "d" double-ptr double mkl_rt)

;; ========================= Sparse Vector engines ============================================

(def ^{:no-doc true :const true} MIXED_UNSUPPORTED_MSG
  "This operation is not supported on mixed sparse and dense vectors.")

(def ^{:no-doc true :const true} SPARSE_UNSUPPORTED_MSG
  "This operation is not supported on sparse.")

(defmacro real-cs-vector-blas* [name t ptr idx-ptr cast blas ones]
  `(extend-type ~name
     Blas
     (swap [this# x# y#]
       (if (indices y#)
         (swap (engine (entries x#)) (entries x#) (entries y#))
         (throw (UnsupportedOperationException. MIXED_UNSUPPORTED_MSG)))
       y#)
     (copy [this# x# y#]
       (if (indices y#)
         (copy (engine (entries x#)) (entries x#) (entries y#))
         (throw (UnsupportedOperationException. MIXED_UNSUPPORTED_MSG)))
       y#)
     (dot [this# x# y#]
       (if (indices y#)
         (dot (engine (entries x#)) (entries x#) (entries y#))
         (. ~blas ~(cblas t 'doti) (dim (entries x#)) (~ptr x#) (~idx-ptr (indices x#)) (~ptr y#))))
     (nrm1 [this# x#]
       (asum this# x#))
     (nrm2 [this# x#]
       (nrm2 (engine (entries x#)) (entries x#)))
     (nrmi [this# x#]
       (amax this# x#))
     (asum [this# x#]
       (asum (engine (entries x#)) (entries x#)))
     (iamax [this# x#]
       (entry (indices x#) (iamax (engine (entries x#)) (entries x#))))
     (iamin [this# x#]
       (entry (indices x#) (iamin (engine (entries x#)) (entries x#))))
     (rot [this# x# y# c# s#]
       (if (indices y#)
         (rot (engine (entries x#)) (entries x#) (entries y#) c# s#)
         (. ~blas ~(cblas t 'roti) (dim (entries x#))
            (~ptr x#) (~idx-ptr (indices x#)) (~ptr y#) (~cast c#) (~cast s#))))
     (rotg [this# abcs#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (rotm [this# x# y# param#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (rotmg [this# d1d2xy# param#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (scal [this# alpha# x#]
       (scal (engine (entries x#)) alpha# (entries x#))
       x#)
     (axpy [this# alpha# x# y#]
       (if (indices y#)
         (axpy (engine (entries x#)) alpha# (entries x#) (entries y#))
         (. ~blas ~(cblas t 'axpyi) (dim (entries x#))
            (~cast alpha#) (~ptr x#) (~idx-ptr (indices x#)) (~ptr y#)))
       y#)
     BlasPlus
     (amax [this# x#]
       (amax (engine (entries x#)) (entries x#)))
     (subcopy [this# x# y# kx# lx# ky#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (sum [this# x#]
       (sum (engine (entries x#)) (entries x#)))
     (imax [this# x#]
       (entry (indices x#) (vector-imax (entries x#))))
     (imin [this# x#]
       (entry (indices x#) (vector-imin (entries x#))))
     (set-all [this# alpha# x#]
       (set-all (engine (entries x#)) alpha# (entries x#))
       x#)
     (axpby [this# alpha# x# beta# y#]
       (if (= 1.0 beta#)
         (axpy this# alpha# x# y#)
         (if (indices y#)
           (axpby (engine (entries x#)) alpha# (entries x#) beta# (entries y#))
           (do (scal (engine y#) beta# (entries y#))
               (axpy this# alpha# x# y#))))
       y#)))

(defmacro real-cs-vector-sparse-blas* [name t ptr idx-ptr blas]
  `(extend-type ~name
     SparseBlas
     (gthr [this# y# x#]
       (. ~blas ~(cblas t 'gthr) (dim (entries x#)) (~ptr y#) (~ptr x#) (~idx-ptr (indices x#)))
       x#)))

(deftype FloatCSVectorEngine [])
(real-cs-vector-blas* FloatCSVectorEngine "s" float-ptr int-ptr float mkl_rt ones-float)
(real-vector-math* FloatCSVectorEngine "s" float-ptr float)
(real-cs-vector-sparse-blas* FloatCSVectorEngine "s" float-ptr int-ptr mkl_rt)

(deftype DoubleCSVectorEngine [])
(real-cs-vector-blas* DoubleCSVectorEngine "d" double-ptr int-ptr double mkl_rt ones-double)
(real-vector-math* DoubleCSVectorEngine "d" double-ptr double)
(real-cs-vector-sparse-blas* DoubleCSVectorEngine "d" double-ptr int-ptr mkl_rt)

(deftype MKLRealFactory [index-fact ^DataAccessor da
                         vector-eng ge-eng cs-vector-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  Factory
  (create-vector [this n init]
    (let-release [res (real-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-ge [this m n column? init]
    (let-release [res (real-ge-matrix this m n column?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  SparseFactory
  (cs-vector-engine [_]
    cs-vector-eng)
  )

(deftype MKLIntegerFactory [index-fact ^DataAccessor da vector-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  Factory
  (create-vector [this n init]
    (let-release [res (integer-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (vector-engine [_]
    vector-eng))

(def float-accessor (->FloatPointerAccessor malloc!))
(def double-accessor (->DoublePointerAccessor malloc!))
(def int-accessor (->IntPointerAccessor malloc!))
(def long-accessor (->LongPointerAccessor malloc!))

(def mkl-int (->MKLIntegerFactory mkl-int int-accessor (->IntVectorEngine)))
(def mkl-long (->MKLIntegerFactory mkl-long long-accessor (->LongVectorEngine)))

(def mkl-float
  (->MKLRealFactory mkl-int float-accessor
                    (->FloatVectorEngine) (->FloatGEEngine) (->FloatCSVectorEngine)))

(def mkl-double
  (->MKLRealFactory mkl-int double-accessor
                    (->DoubleVectorEngine) (->DoubleGEEngine) (->DoubleCSVectorEngine)))
