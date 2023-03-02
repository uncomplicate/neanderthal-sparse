(ns uncomplicate.neanderthal.internal.cpp.mkl.factory
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release]]
             [utils :refer [dragan-says-ex generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojure-cpp :refer [long-pointer float-pointer double-pointer put!]]
            [uncomplicate.neanderthal
             [core :refer [dim entry]]
             [math :refer [f=] :as math]
             [block :refer [create-data-source initialize buffer offset stride]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage]]
             [common :refer [check-stride check-eq-navigators real-accessor]]]
            [uncomplicate.neanderthal.internal.cpp
             [structures :refer :all]
             [lapack :refer :all]
             [blas :refer [float-ptr double-ptr vector-imax vector-imin]]]
            [uncomplicate.neanderthal.internal.cpp.mkl.core :refer [malloc!]]
            [uncomplicate.neanderthal.internal.host.mkl
             :refer [sigmoid-over-tanh vector-ramp vector-relu vector-elu]])
  (:import [uncomplicate.neanderthal.internal.api DataAccessor Block Vector]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$VSLStreamStatePtr]))

;; =============== Factories ==================================================

(def ^:const mkl-blas-layout
  {:row mkl_rt/CblasRowMajor
   :column mkl_rt/CblasColMajor})

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

(defmacro real-vector-blas* [name t ptr cast blas lapack blas-layout ones]
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
       (. ~blas ~(cblas t 'rot) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~cast c#) (~cast s#)))
     (rotg [this# abcs#]
       (check-stride abcs#)
       (. ~blas ~(cblas t 'rotg) (~ptr abcs#) (~ptr abcs# 1) (~ptr abcs# 2) (~ptr abcs# 3))
       abcs#)
     (rotm [this# x# y# param#]
       (. ~blas ~(cblas t 'rotm) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr param#)))
     (rotmg [this# d1d2xy# param#]
       (check-stride d1d2xy# param#)
       (. ~blas ~(cblas t 'rotmg) (~ptr d1d2xy#) (~ptr d1d2xy# 1) (~ptr d1d2xy# 2) (~cast (entry d1d2xy# 3)) (~ptr param#)))
     (scal [this# alpha# x#]
       (. ~blas ~(cblas t 'scal) (dim x#) (~cast alpha#) (~ptr x#) (stride x#))
       x#)
     (axpy [this# alpha# x# y#]
       (. ~blas ~(cblas t 'axpy) (dim x#) (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     BlasPlus
     (amax [this# x#]
       (vector-amax x#))
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
         (. ~lapack ~(lapacke t 'laset) (int ~(:row blas-layout)) ~(byte (int \g)) (dim x#) 1
            (~cast alpha#) (~cast alpha#) (~ptr x#) (stride x#)))
       x#)
     (axpby [this# alpha# x# beta# y#] ;; TODO axpby will be available in JavaCPP 1.5.9
       (. ~blas ~(cblas t 'scal) (dim y#) (~cast beta#) (~ptr y#) (stride y#))
       (. ~blas ~(cblas t 'axpy) (dim x#) (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
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

(defn create-stream-ars5 ^mkl_rt$VSLStreamStatePtr [seed]
  (let-release [stream (mkl_rt$VSLStreamStatePtr. (long-pointer 1))]
    (with-mkl-check (mkl_rt/vslNewStream stream mkl_rt/VSL_BRNG_ARS5 seed)
      stream)))

(def ^:private default-rng-stream (create-stream-ars5 (generate-seed)))

(defmacro real-vector-math* [name t ptr cast]
  `(extend-type ~name
     VectorMath
     (sqr [this# a# y#]
       (vector-math ~(math t 'Sqr) ~ptr a# y#))
     (mul [this# a# b# y#]
       (vector-math ~(math t 'Mul) ~ptr a# b# y#))
     (div [this# a# b# y#]
       (vector-math ~(math t 'Div) ~ptr a# b# y#))
     (inv [this# a# y#]
       (vector-math ~(math t 'Inv) ~ptr a# y#))
     (abs [this# a# y#]
       (vector-math ~(math t 'Abs) ~ptr a# y#))
     (linear-frac [this# a# b# scalea# shifta# scaleb# shiftb# y#]
       (check-stride a# b# y#)
       (. mkl_rt ~(math t 'LinearFrac) (dim a#) (~ptr a#) (~ptr b#)
          (~cast scalea#) (~cast shifta#) (~cast scaleb#) (~cast shiftb#) (~ptr y#))
       y#)
     (fmod [this# a# b# y#]
       (vector-math ~(math t 'Fmod) ~ptr a# b# y#))
     (frem [this# a# b# y#]
       (vector-math  ~(math t 'Remainder) ~ptr a# b# y#))
     (sqrt [this# a# y#]
       (vector-math ~(math t 'Sqrt) ~ptr a# y#))
     (inv-sqrt [this# a# y#]
       (vector-math ~(math t 'InvSqrt) ~ptr a# y#))
     (cbrt [this# a# y#]
       (vector-math ~(math t 'Cbrt) ~ptr a# y#))
     (inv-cbrt [this# a# y#]
       (vector-math ~(math t 'InvCbrt) ~ptr a# y#))
     (pow2o3 [this# a# y#]
       (vector-math ~(math t 'Pow2o3) ~ptr a# y#))
     (pow3o2 [this# a# y#]
       (vector-math ~(math t 'Pow3o2) ~ptr a# y#))
     (pow [this# a# b# y#]
       (vector-math ~(math t 'Pow) ~ptr a# b# y#))
     (powx [this# a# b# y#]
       (check-stride a# y#)
       (. mkl_rt ~(math t 'Powx) (dim a#) (~ptr a#) (~cast b#) (~ptr y#))
       y#)
     (hypot [this# a# b# y#]
       (vector-math ~(math t 'Hypot) ~ptr a# b# y#))
     (exp [this# a# y#]
       (vector-math ~(math t 'Exp) ~ptr a# y#))
     (exp2 [this# a# y#]
       (vector-math ~(math t 'Exp2) ~ptr a# y#))
     (exp10 [this# a# y#]
       (vector-math ~(math t 'Exp10) ~ptr a# y#))
     (expm1 [this# a# y#]
       (vector-math ~(math t 'Expm1) ~ptr a# y#))
     (log [this# a# y#]
       (vector-math ~(math t 'Ln) ~ptr a# y#))
     (log2 [this# a# y#]
       (vector-math ~(math t 'Log2) ~ptr a# y#))
     (log10 [this# a# y#]
       (vector-math ~(math t 'Log10) ~ptr a# y#))
     (log1p [this# a# y#]
       (vector-math ~(math t 'Log1p) ~ptr a# y#))
     (sin [this# a# y#]
       (vector-math ~(math t 'Sin) ~ptr a# y#))
     (cos [this# a# y#]
       (vector-math ~(math t 'Cos) ~ptr a# y#))
     (tan [this# a# y#]
       (vector-math ~(math t 'Tan) ~ptr a# y#))
     (sincos [this# a# y# z#]
       (vector-math ~(math t 'SinCos) ~ptr a# y# z#))
     (asin [this# a# y#]
       (vector-math ~(math t 'Asin) ~ptr a# y#))
     (acos [this# a# y#]
       (vector-math ~(math t 'Acos) ~ptr a# y#))
     (atan [this# a# y#]
       (vector-math ~(math t 'Atan) ~ptr a# y#))
     (atan2 [this# a# b# y#]
       (vector-math ~(math t 'Atan2) ~ptr a# b# y#))
     (sinh [this# a# y#]
       (vector-math ~(math t 'Sinh) ~ptr a# y#))
     (cosh [this# a# y#]
       (vector-math ~(math t 'Cosh) ~ptr a# y#))
     (tanh [this# a# y#]
       (vector-math ~(math t 'Tanh) ~ptr a# y#))
     (asinh [this# a# y#]
       (vector-math ~(math t 'Asinh) ~ptr a# y#))
     (acosh [this# a# y#]
       (vector-math ~(math t 'Acosh) ~ptr a# y#))
     (atanh [this# a# y#]
       (vector-math ~(math t 'Atanh) ~ptr a# y#))
     (erf [this# a# y#]
       (vector-math ~(math t 'Erf) ~ptr a# y#))
     (erfc [this# a# y#]
       (vector-math ~(math t 'Erfc) ~ptr a# y#))
     (erf-inv [this# a# y#]
       (vector-math ~(math t 'ErfInv) ~ptr a# y#))
     (erfc-inv [this# a# y#]
       (vector-math ~(math t 'ErfcInv) ~ptr a# y#))
     (cdf-norm [this# a# y#]
       (vector-math ~(math t 'CdfNorm) ~ptr a# y#))
     (cdf-norm-inv [this# a# y#]
       (vector-math ~(math t 'CdfNormInv) ~ptr a# y#))
     (gamma [this# a# y#]
       (vector-math ~(math t 'TGamma) ~ptr a# y#))
     (lgamma [this# a# y#]
       (vector-math ~(math t 'LGamma) ~ptr a# y#))
     (expint1 [this# a# y#]
       (vector-math ~(math t 'ExpInt1) ~ptr a# y#))
     (floor [this# a# y#]
       (vector-math ~(math t 'Floor) ~ptr a# y#))
     (fceil [this# a# y#]
       (vector-math ~(math t 'Ceil) ~ptr a# y#))
     (trunc [this# a# y#]
       (vector-math ~(math t 'Trunc) ~ptr a# y#))
     (round [this# a# y#]
       (vector-math ~(math t 'Round) ~ptr a# y#))
     (modf [this# a# y# z#]
       (vector-math ~(math t 'Modf) ~ptr a# y# z#))
     (frac [this# a# y#]
       (vector-math ~(math t 'Frac) ~ptr a# y#))
     (fmin [this# a# b# y#]
       (vector-math ~(math t 'Fmin) ~ptr a# b# y#))
     (fmax [this# a# b# y#]
       (vector-math ~(math t 'Fmax) ~ptr a# b# y#))
     (copy-sign [this# a# b# y#]
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
                                             (doto (float-pointer 1) (put! 0 1.0)) 1 0 0))
(def ^:private ones-double (->RealBlockVector nil nil nil true
                                              (doto (double-pointer 1) (put! 0 1.0)) 1 0 0))

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt mkl-blas-layout ones-float)
(real-vector-math* FloatVectorEngine "s" float-ptr float)

(extend-type FloatVectorEngine
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (float-uniform (or rng-stream default-rng-stream) lower upper x))
  (rand-normal [this rng-stream mu sigma x]
    (with-rng-check x
      (float-gaussian (or rng-stream default-rng-stream) mu sigma x))))

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt mkl-blas-layout ones-double)
(real-vector-math* DoubleVectorEngine "d" double-ptr double)

(extend-type DoubleVectorEngine
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (double-uniform (or rng-stream default-rng-stream) lower upper x))
  (rand-normal [this rng-stream mu sigma x]
    (with-rng-check x
      (double-gaussian (or rng-stream default-rng-stream) mu sigma x))))

(deftype MKLRealFactory [index-fact ^DataAccessor da
                         vector-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    @index-fact)
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
  (vector-engine [_]
    vector-eng)
  )

(def float-accessor (->FloatPointerAccessor malloc!))
(def double-accessor (->DoublePointerAccessor malloc!))

(def mkl-float
  (->MKLRealFactory mkl-int float-accessor
                    (->FloatVectorEngine)))

(def mkl-double
  (->MKLRealFactory mkl-int double-accessor
                    (->DoubleVectorEngine)))
