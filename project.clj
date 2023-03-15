;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/neanderthal-sparse "0.1.0-SNAPSHOT"
  :description "Fast Clojure Sparse Matrix Library"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/neanderthal-sparse"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.uncomplicate/clojure-cpp "0.1.0-SNAPSHOT"]
                 [uncomplicate/neanderthal "0.46.0"]
                 [org.bytedeco/mkl "2022.2-1.5.9-SNAPSHOT"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.7"]]
                   :resource-paths ["data"]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.10.9"]
                                  [org.bytedeco/mkl-platform "2022.2-1.5.9-SNAPSHOT"]]

                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"
                                        "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
                                        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]
  :codox {:metadata {:doc/format :markdown}
          :src-dir-uri "http://github.com/uncomplicate/neanderthal-sparse/blob/master/"
          :src-linenum-anchor-prefix "L"
          :output-path "docs/codox"}

  :source-paths ["src/clojure" "src/device"])
