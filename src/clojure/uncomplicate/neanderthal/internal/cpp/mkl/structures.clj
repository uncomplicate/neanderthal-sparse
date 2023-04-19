;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.structures
  (:require
   [uncomplicate.commons
    [core :refer [Releaseable release let-release Info info Viewable view]]
    [utils :refer [dragan-says-ex]]]
   [uncomplicate.fluokitten.protocols
    :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative fold]]
   [uncomplicate.clojure-cpp :refer [pointer pointer-seq fill! capacity! byte-buffer float-pointer
                                     double-pointer long-pointer int-pointer short-pointer byte-pointer
                                     element-count null?]]
   [uncomplicate.neanderthal
    [core :refer [dim transfer! mrows ncols]]
    [block :refer [entry-type offset stride buffer column?]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [navigation :refer :all]
    [printing :refer [print-vector]]]
   [uncomplicate.neanderthal.internal.host.fluokitten :refer :all]
   [uncomplicate.neanderthal.internal.cpp.structures
    :refer [CompressedSparse entries indices vector-seq csr-engine CSR]]
   [uncomplicate.neanderthal.internal.cpp.mkl.core :refer [create-csr matrix-descr]])
  (:import [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL IFn$LLLL]
           org.bytedeco.mkl.global.mkl_rt
           [uncomplicate.neanderthal.internal.api Block Matrix DataAccessor RealNativeMatrix
            IntegerVector LayoutNavigator MatrixImplementation RealAccessor IntegerAccessor]))

(declare csr-matrix)

;; ======================= Compressed Sparse Matrix ======================================

(deftype CSRMatrix [^LayoutNavigator nav fact eng spm desc
                    ^Block nzx ^IntegerVector indx ^IntegerVector pb ^IntegerVector pe
                    ^long m ^long n]
  Object
  (hashCode [_]
    (-> (hash :CSRMatrix) (hash-combine nzx) (hash-combine indx) (hash-combine pb) (hash-combine pe)))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (instance? CSRMatrix b)
      (and (= m (mrows b)) (= n (ncols b))
           (= nzx (entries b)) (= indx (indices b)) (= pb (.pb ^CSRMatrix b)) (= pe (.pe ^CSRMatrix b)))
      :default false))
  (toString [_]
    (format "#CSRMatrix[%s, mxn:%dx%d, layout%s]"
            (entry-type (data-accessor nzx)) m n (dec-property (.layout nav))))
  MatrixImplementation
  (matrixType [_]
    :cs)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  Info
  (info [x]
    {:entry-type (.entryType (data-accessor nzx))
     :class (class x)
     :device (info nzx :device)
     :dim n
     :engine (info eng)})
  (info [x info-type]
    (case info-type
      :entry-type (.entryType (data-accessor nzx))
      :class (class x)
      :device (info nzx :device)
      :dim n
      :engine (info eng)
      nil))
  Releaseable
  (release [_]
    (release spm)
    (release desc)
    (release nzx)
    (release indx)
    (release pb)
    (release pe)
    true)
  Seqable
  (seq [x]
    (vector-seq nzx))
  MemoryContext
  (compatible? [_ b]
    (and (compatible? nzx (entries b))))
  (fits? [_ b]
    (and (instance? CSRMatrix b) (fits? nzx (entries b)) (fits? indx (indices b))
         (fits? pe (.pe ^CSRMatrix b)) (fits? pb (.pb ^CSRMatrix b)))) ;; TODO region?
  (device [_]
    (device nzx))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor fact))
  Container
  (raw [a]
    (raw a fact))
  (raw [_ fact]
    (csr-matrix (factory fact) m n (view indx) (view pb) (view pe) (view desc) false))
  (zero [a]
    (zero a fact))
  (zero [_ fact]
    (csr-matrix (factory fact) m n (view indx) (view pb) (view pe) (view desc) true)) ;;TODO indices on the gpu etc.
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)
      res))
  (native [a]
    a)
  Viewable
  (view [a]
    (csr-matrix m n (view indx) (view pb) (view pe) (view nzx) nav (view desc)))
  Block
  (buffer [_]
    (.buffer nzx))
  (offset [_]
    (.offset nzx))
  (stride [_]
    (.stride nzx))
  (isContiguous [_]
    (.isContiguous ^Block nzx))
  Matrix
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (row [a i]
    #_(cs-vector false buf n (+ ofst (.index nav stor i 0)) ;;TODO only the compressed stripe available (row/col)
                 (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    #_(sparse-rows a)) ;;TODO
  (col [a j]
    #_(cs-vector fact false buf m (+ ofst (.index nav stor 0 j)) ;;TODO
                 (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    #_(sparse-cols a)) ;;TODO
  (dia [a]
    #_(dragan-says-ex "Diagonal of a sparse matrix is not available in a general case")) ;; TODO only if the matrix is diagonal.
  (dia [a k]
    #_(dragan-says-ex "Diagonal of a sparse matrix is not available in a general case")) ;; TODO only if the matrix is diagonal.
  (dias [a]
    #_(sparse-dias a)) ;;TODO
  (submatrix [a i j k l]
    #_(csr-matrix fact false buf k l (+ ofst (.index nav stor i j)) ;; TODO see if it's possible in general case
                  nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    #_(csr-matrix fact false buf n m ofst (flip nav) stor (flip reg))) ;; TODO
  CompressedSparse
  (entries [_]
    nzx)
  (indices [_]
    indx)
  CSR
  (columns [_]
    indx)
  (indexb [_]
    pb)
  (indexe [_]
    pe))

(defn csr-matrix
  ([m n indx pb pe nzx ^LayoutNavigator nav desc]
   (let [fact (factory nzx)]
     (if (compatible? (index-factory nzx) indx)
       (if (and (<= 0 (dim indx) (dim nzx)) (<= (long (if (.isColumnMajor nav) n m)) (dim pe) (dim pb))
                (= 1 (stride indx) (stride nzx)) (= 0 (offset indx) (offset nzx)) (fits? indx nzx)) ;;TODO better error message. Perhaps automatize (.isColumnmajor part)
         (->CSRMatrix nav fact (csr-engine fact)
                      (create-csr (buffer nzx) 0 (max 1 (long m)) (max 1 (long n))
                                  (buffer pb) (buffer pe) (buffer indx))
                      desc nzx indx pb pe m n)
         (dragan-says-ex "Non-zero vector and index vector have to fit each other." {:nzx nzx :indx indx}));;TODO error message
       (dragan-says-ex "Incompatible index vector." {:required (index-factory nzx) :provided (factory indx)}))))
  ([fact m n indx pb pe nav desc init]
   (let-release [nzx (create-vector fact (dim indx) init)]
     (csr-matrix m n indx pb pe nzx nav desc))))

(defn ge-csr-matrix
  ([fact m n indx pb pe column? init]
   (csr-matrix fact m n indx pb pe (layout-navigator column?) (matrix-descr :ge) init))
  ([fact m n indx pb pe column?]
   (csr-matrix fact m n indx pb pe column? true)))

(defmethod print-method CSRMatrix [^CSRMatrix x ^java.io.Writer w] ;; TODO transform to nested vectors
  (.write w (format "%s\n%s" (str x) (pr-str (seq (indices x)))))
  (when-not (null? (buffer (entries x)))
    (print-vector w (entries x))))

(defmethod transfer! [CSRMatrix CSRMatrix]
  [source destination]
  (transfer! (entries source) (entries destination))
  destination)

(defn seq-to-csr [source]
  (if (number? (get-in source [0 0]))
    (seq-to-csr (partition 2 source))
    (reduce (fn [[^long row ptrs idx vals] [ridx rvals]]
              (let [fill (- (count ridx) (count rvals))]
                (if (<= 0 fill)
                  [(inc row)
                   (conj ptrs (+ (long (peek ptrs)) (long (count ridx))))
                   (into idx ridx)
                   (-> vals (into rvals) (into (repeat fill 0.0)))]
                  (dragan-says-ex "Each value of a sparse matrix need its row/col position."
                                  {:row row :idx-count (count ridx) :val-count (count rvals)}))))
            [0 [0] [] []]
            source)))

(defmethod transfer! [clojure.lang.Sequential CSRMatrix]
  [source destination]
  (transfer! (seq-to-csr 3) (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[D") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[F") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[J") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[I") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [CSRMatrix (Class/forName "[D")]
  [source destination]
  (transfer! (entries source) destination))

(defmethod transfer! [CSRMatrix (Class/forName "[F")]
  [source destination]
  (transfer! (entries source) destination))

;;TODO handle heterogenous types (float/double...)
#_(defmethod transfer! [RealBlockVector CSMatrix]
  [^RealBlockVector source ^CSMatrix destination]
  (gthr (engine destination) source destination)
  destination)
