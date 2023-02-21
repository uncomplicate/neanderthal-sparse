(ns uncomplicate.neanderthal.internal.cpp.mkl.factory
     (:require [uncomplicate.commons
              [core :refer [with-release let-release info Releaseable release]]
              [utils :refer [dragan-says-ex generate-seed direct-buffer]]]
             [uncomplicate.fluokitten.core :refer [fmap!]]
             [uncomplicate.neanderthal
              [core :refer [dim]]
              [math :refer [f=] :as math]
              [block :refer [create-data-source initialize]]]
             [uncomplicate.neanderthal.internal
              [api :refer :all]
              [navigation :refer [full-storage]]
              [common :refer [check-stride check-eq-navigators real-accessor]]]
             [uncomplicate.neanderthal.internal.cpp
              [structures :refer :all]]
             [uncomplicate.neanderthal.internal.cpp.mkl.core :refer [malloc!]])
     (:import [uncomplicate.neanderthal.internal.api DataAccessor Block]))

;; =============== Factories ==================================================

(declare mkl-int)

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
  (->MKLRealFactory mkl-int float-accessor nil))
