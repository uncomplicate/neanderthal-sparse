(ns uncomplicate.neanderthal.internal.cpp.mkl.factory
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release]]
             [utils :refer [dragan-says-ex generate-seed direct-buffer]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
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

(defmacro real-vector-blas* [name t ptr cast blas lapack blas-layout]
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
     ;; (rotg [this# abcs] TODO
     ;;   (mkl_rt/cblas_srotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
     ;;   abcs)
     (rotm [this# x# y# param#]
       (.~blas ~(cblas t 'rotm) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr param#)))
     ;; (rotmg [this# d1d2xy param]
     ;;   (vector-rotmg mkl_rt/cblas_srotmg d1d2xy param))
     (scal [this# alpha# x#]
       (. ~blas ~(cblas t 'scal) (dim x#) (~cast alpha#) (~ptr x#) (stride x#))
       x#)
     (axpy [this# alpha# x# y#]
       (. ~blas ~(cblas t 'axpy) (dim x#) (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     BlasPlus
     (amax [this# x#]
       (vector-amax x#))
     ;; (subcopy [this# x y kx lx ky]
     ;;   (vector-subcopy mkl_rt/cblas_scopy lx (.buffer ^RealBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
     ;;                (.buffer ^RealBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
     ;;   y)
     ;; (sum [this# x]
     ;;   (vector-sum CBLAS/sdot ^RealBlockVector x ^RealBlockVector ones-float))
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

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt mkl-blas-layout)

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt mkl-blas-layout)

(def mkl-float
  (->MKLRealFactory mkl-int float-accessor
                    (->FloatVectorEngine)))

(def mkl-double
  (->MKLRealFactory mkl-int double-accessor
                    (->DoubleVectorEngine)))
