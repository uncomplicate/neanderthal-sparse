(ns uncomplicate.neanderthal.internal.cpp.mkl.factory
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release]]
             [utils :refer [dragan-says-ex generate-seed direct-buffer]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojure-cpp :refer [float-pointer]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
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
            [uncomplicate.neanderthal.internal.cpp.mkl.core :refer [malloc!]])
  (:import [uncomplicate.neanderthal.internal.api DataAccessor Block Vector]
           [org.bytedeco.mkl.global mkl_rt]))

;; =============== Factories ==================================================

(declare mkl-int)


(deftype FloatVectorEngine []
  Blas
  (swap [_ x y]
    (mkl_rt/cblas_sswap (dim x) (float-ptr x) (stride x) (float-ptr y) (stride y))
    x)
  (copy [_ x y]
    (mkl_rt/cblas_scopy (dim x) (float-ptr x) (stride x) (float-ptr y) (stride y))
    y)
  (dot [_ x y]
    (mkl_rt/cblas_sdot (dim x) (float-ptr x) (stride x) (float-ptr y) (stride y)))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (mkl_rt/cblas_snrm2 (dim x) (float-ptr x) (stride x)))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (mkl_rt/cblas_sasum (dim x) (float-ptr x) (stride x)))
  (iamax [_ x]
    (mkl_rt/cblas_isamax (dim x) (float-ptr x) (stride x)))
  (iamin [_ x]
    (mkl_rt/cblas_isamin (dim x) (float-ptr x) (stride x)))
  (rot [_ x y c s]
    (mkl_rt/cblas_srot (dim x) (float-ptr x) (stride x) (float-ptr y) (stride y) (float c) (float s)))
  ;; (rotg [_ abcs] TODO
  ;;   (mkl_rt/cblas_srotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
  ;;   abcs)
  (rotm [_ x y param]
    (mkl_rt/cblas_srotm (dim x) (float-ptr x) (stride x) (float-ptr y) (stride y) (float-ptr param)))
  ;; (rotmg [_ d1d2xy param]
  ;;   (vector-rotmg mkl_rt/cblas_srotmg d1d2xy param))
  (scal [_ alpha x]
    (mkl_rt/cblas_sscal (dim x) (float alpha) (float-ptr x) (stride x))
    x)
  (axpy [_ alpha x y]
    (mkl_rt/cblas_saxpy (dim x) (float alpha) (float-ptr x) (stride x) (float-ptr y) (stride y))
    y)
  BlasPlus
  (amax [_ x]
    (vector-amax x))
  ;; (subcopy [_ x y kx lx ky]
  ;;   (vector-subcopy mkl_rt/cblas_scopy lx (.buffer ^RealBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
  ;;                (.buffer ^RealBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
  ;;   y)
  ;; (sum [_ x]
  ;;   (vector-sum CBLAS/sdot ^RealBlockVector x ^RealBlockVector ones-float))
  (imax [_ x]
    (vector-imax x))
  (imin [_ x]
    (vector-imin x))
  (set-all [_ alpha x]
    (vctr-laset mkl_rt/LAPACKE_slaset float-ptr (float alpha) x))
  (axpby [_ alpha x beta y]
    (mkl_rt/cblas_sscal (dim y) (float beta) (float-ptr y) (stride y))
    (mkl_rt/cblas_saxpy (dim x) (float alpha) (float-ptr x) (stride x)
                        (float-ptr y) (stride y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt mkl_rt/LAPACKE_slasrt float-ptr x increasing))
  ;; VectorMath
  ;; (sqr [_ a y]
  ;;   (vector-math MKL/vsSqr ^RealBlockVector a ^RealBlockVector y))
  ;; (mul [_ a b y]
  ;;   (vector-math MKL/vsMul ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (div [_ a b y]
  ;;   (vector-math MKL/vsDiv ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (inv [_ a y]
  ;;   (vector-math MKL/vsInv ^RealBlockVector a ^RealBlockVector y))
  ;; (abs [_ a y]
  ;;   (vector-math MKL/vsAbs ^RealBlockVector a ^RealBlockVector y))
  ;; (linear-frac [_ a b scalea shifta scaleb shiftb y]
  ;;   (vector-linear-frac MKL/vsLinearFrac ^RealBlockVector a ^RealBlockVector b
  ;;                       scalea shifta scaleb shiftb ^RealBlockVector y))
  ;; (fmod [_ a b y]
  ;;   (vector-math MKL/vsFmod ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (frem [_ a b y]
  ;;   (vector-math MKL/vsRemainder ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (sqrt [_ a y]
  ;;   (vector-math MKL/vsSqrt ^RealBlockVector a ^RealBlockVector y))
  ;; (inv-sqrt [_ a y]
  ;;   (vector-math MKL/vsInvSqrt ^RealBlockVector a ^RealBlockVector y))
  ;; (cbrt [_ a y]
  ;;   (vector-math MKL/vsCbrt ^RealBlockVector a ^RealBlockVector y))
  ;; (inv-cbrt [_ a y]
  ;;   (vector-math MKL/vsInvCbrt ^RealBlockVector a ^RealBlockVector y))
  ;; (pow2o3 [_ a y]
  ;;   (vector-math MKL/vsPow2o3 ^RealBlockVector a ^RealBlockVector y))
  ;; (pow3o2 [_ a y]
  ;;   (vector-math MKL/vsPow3o2 ^RealBlockVector a ^RealBlockVector y))
  ;; (pow [_ a b y]
  ;;   (vector-math MKL/vsPow ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (powx [_ a b y]
  ;;   (vector-powx MKL/vsPowx ^RealBlockVector a b ^RealBlockVector y))
  ;; (hypot [_ a b y]
  ;;   (vector-math MKL/vsHypot ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (exp [_ a y]
  ;;   (vector-math MKL/vsExp ^RealBlockVector a ^RealBlockVector y))
  ;; (exp2 [_ a y]
  ;;   (vector-math MKL/vsExp2 ^RealBlockVector a ^RealBlockVector y))
  ;; (exp10 [_ a y]
  ;;   (vector-math MKL/vsExp10 ^RealBlockVector a ^RealBlockVector y))
  ;; (expm1 [_ a y]
  ;;   (vector-math MKL/vsExpm1 ^RealBlockVector a ^RealBlockVector y))
  ;; (log [_ a y]
  ;;   (vector-math MKL/vsLn ^RealBlockVector a ^RealBlockVector y))
  ;; (log2 [_ a y]
  ;;   (vector-math MKL/vsLog2 ^RealBlockVector a ^RealBlockVector y))
  ;; (log10 [_ a y]
  ;;   (vector-math MKL/vsLog10 ^RealBlockVector a ^RealBlockVector y))
  ;; (log1p [_ a y]
  ;;   (vector-math MKL/vsLog1p ^RealBlockVector a ^RealBlockVector y))
  ;; (sin [_ a y]
  ;;   (vector-math MKL/vsSin ^RealBlockVector a ^RealBlockVector y))
  ;; (cos [_ a y]
  ;;   (vector-math MKL/vsCos ^RealBlockVector a ^RealBlockVector y))
  ;; (tan [_ a y]
  ;;   (vector-math MKL/vsTan ^RealBlockVector a ^RealBlockVector y))
  ;; (sincos [_ a y z]
  ;;   (vector-math MKL/vsSinCos ^RealBlockVector a ^RealBlockVector y ^RealBlockVector z))
  ;; (asin [_ a y]
  ;;   (vector-math MKL/vsAsin ^RealBlockVector a ^RealBlockVector y))
  ;; (acos [_ a y]
  ;;   (vector-math MKL/vsAcos ^RealBlockVector a ^RealBlockVector y))
  ;; (atan [_ a y]
  ;;   (vector-math MKL/vsAtan ^RealBlockVector a ^RealBlockVector y))
  ;; (atan2 [_ a b y]
  ;;   (vector-math MKL/vsAtan2 ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (sinh [_ a y]
  ;;   (vector-math MKL/vsSinh ^RealBlockVector a ^RealBlockVector y))
  ;; (cosh [_ a y]
  ;;   (vector-math MKL/vsCosh ^RealBlockVector a ^RealBlockVector y))
  ;; (tanh [_ a y]
  ;;   (vector-math MKL/vsTanh ^RealBlockVector a ^RealBlockVector y))
  ;; (asinh [_ a y]
  ;;   (vector-math MKL/vsAsinh ^RealBlockVector a ^RealBlockVector y))
  ;; (acosh [_ a y]
  ;;   (vector-math MKL/vsAcosh ^RealBlockVector a ^RealBlockVector y))
  ;; (atanh [_ a y]
  ;;   (vector-math MKL/vsAtanh ^RealBlockVector a ^RealBlockVector y))
  ;; (erf [_ a y]
  ;;   (vector-math MKL/vsErf ^RealBlockVector a ^RealBlockVector y))
  ;; (erfc [_ a y]
  ;;   (vector-math MKL/vsErfc ^RealBlockVector a ^RealBlockVector y))
  ;; (erf-inv [_ a y]
  ;;   (vector-math MKL/vsErfInv ^RealBlockVector a ^RealBlockVector y))
  ;; (erfc-inv [_ a y]
  ;;   (vector-math MKL/vsErfcInv ^RealBlockVector a ^RealBlockVector y))
  ;; (cdf-norm [_ a y]
  ;;   (vector-math MKL/vsCdfNorm ^RealBlockVector a ^RealBlockVector y))
  ;; (cdf-norm-inv [_ a y]
  ;;   (vector-math MKL/vsCdfNormInv ^RealBlockVector a ^RealBlockVector y))
  ;; (gamma [_ a y]
  ;;   (vector-math MKL/vsGamma ^RealBlockVector a ^RealBlockVector y))
  ;; (lgamma [_ a y]
  ;;   (vector-math MKL/vsLGamma ^RealBlockVector a ^RealBlockVector y))
  ;; (expint1 [_ a y]
  ;;   (vector-math MKL/vsExpInt1 ^RealBlockVector a ^RealBlockVector y))
  ;; (floor [_ a y]
  ;;   (vector-math MKL/vsFloor ^RealBlockVector a ^RealBlockVector y))
  ;; (fceil [_ a y]
  ;;   (vector-math MKL/vsCeil ^RealBlockVector a ^RealBlockVector y))
  ;; (trunc [_ a y]
  ;;   (vector-math MKL/vsTrunc ^RealBlockVector a ^RealBlockVector y))
  ;; (round [_ a y]
  ;;   (vector-math MKL/vsRound ^RealBlockVector a ^RealBlockVector y))
  ;; (modf [_ a y z]
  ;;   (vector-math MKL/vsModf ^RealBlockVector a ^RealBlockVector y ^RealBlockVector z))
  ;; (frac [_ a y]
  ;;   (vector-math MKL/vsFrac ^RealBlockVector a ^RealBlockVector y))
  ;; (fmin [_ a b y]
  ;;   (vector-math MKL/vsFmin ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (fmax [_ a b y]
  ;;   (vector-math MKL/vsFmax ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (copy-sign [_ a b y]
  ;;   (vector-math MKL/vsCopySign ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  ;; (sigmoid [this a y]
  ;;   (sigmoid-over-tanh this a y))
  ;; (ramp [this a y]
  ;;   (vector-ramp this a y))
  ;; (relu [this alpha a y]
  ;;   (vector-relu this alpha a y))
  ;; (elu [this alpha a y]
  ;;   (vector-elu this alpha a y))
  ;; RandomNumberGenerator
  ;; (rand-uniform [_ rng-stream lower upper x]
  ;;   (vector-random MKL/vsRngUniform (or rng-stream default-rng-stream)
  ;;                  lower upper ^RealBlockVector x))
  ;; (rand-normal [_ rng-stream mu sigma x]
  ;;   (vector-random MKL/vsRngGaussian (or rng-stream default-rng-stream)
  ;;                  mu sigma ^RealBlockVector x))
  )

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
    #_(create-stream-ars5 seed));;TODO
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
  (->MKLRealFactory mkl-int float-accessor (->FloatVectorEngine)))
