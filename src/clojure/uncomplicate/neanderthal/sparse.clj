;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.sparse
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release view]]
             [utils :refer [dragan-says-ex direct-buffer]]]
            [uncomplicate.neanderthal
             [core :refer [vctr? vctr transfer! copy!]]]
            [uncomplicate.neanderthal.internal
             [api :as api]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.structures :refer [cs-vector]]))

(defn csv
  "TODO"
  ([factory n idx nz & nzs]
   (let [idx-factory (api/index-factory factory)]
     (let-release [res (csv factory n idx)]
       (if-not nzs
         (transfer! nz res)
         (transfer! (cons nz nzs) res)))))
  ([factory ^long n source]
   (let [idx-factory (api/index-factory factory)]
     (if (vctr? source)
       (let-release [idx (if (compatible? idx-factory source)
                           (view source)
                           (vctr idx-factory source))]
         (cs-vector (api/factory factory) n idx true))
       (let [[s0 s1] source]
         (if (sequential? s0)
           (if (sequential? s1)
             (csv factory n s0 s1)
             (csv factory n s0 (rest source)))
           (if (and (integer? s0) (integer? s1))
             (csv factory n (vctr idx-factory source))
             (csv factory n (take-nth 2 source) (take-nth 2 (rest source))))))))))

#_(defn cs
  "TODO"
  ([factory m n source options]
   (if (and (<= 0 (long m)) (<= 0 (long n)))
     (let-release [res (api/create-cs (api/factory factory) m n (api/options-column? options)
                                      (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "Compressed sparse matrix cannot have a negative dimension." {:m m :n n})))
  ([factory ^long m ^long n arg]
   (if (or (not arg) (map? arg))
     (cs factory m n nil arg)
     (cs factory m n arg nil)))
  ([factory ^long m ^long n]
   (cs factory m n nil nil))
  ([factory a]
   (let-release [res (transfer (api/factory factory) a)]
     (if (matrix? res)
       res
       (dragan-says-ex "This is not a valid source for matrices.")))))
