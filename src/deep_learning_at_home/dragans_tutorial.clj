;; The code in here comes from the **truly amazing** tutorial series written by Dragan Djuric:

;; Deep Learning in Clojure from Scratch to GPU
;; https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-0-Why-Bother

;; As the articles contain code, I wanted to try it out, and this is how this file grew,
;; article by article, as my notes in a single ns. The code commented out with #_ takes
;; time to eval so is commented out to speed up ns loading and can be eval'd by hand.

(ns deep-learning-at-home.dragans-tutorial
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Releaseable release]]
             [utils :refer [direct-buffer]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal.internal.api :refer [native-factory device]]
            [uncomplicate.neanderthal
             [native :refer [dv dge fge native-double native-float fv]]
             [core :refer :all]
             [math :as math :refer [signum #_exp sqr sqrt log pi sqr]]
             [real :as real]
             [vect-math :refer [fmax! tanh! linear-frac! mul mul! exp sin cos exp! cosh! inv! sqr!]]
             #_[cuda :refer [cuda-float]]
             [opencl :refer [opencl-float]]
             [block :refer [buffer]]
             [random :as random]]
    #_[uncomplicate.clojurecuda.core :as cuda :refer [current-context default-stream synchronize!]]
            [uncomplicate.clojurecl.core :as opencl :refer [*context* *command-queue* finish!]]
            [criterium.core :refer [quick-bench]])
  (:import (java.math RoundingMode)
           (clojure.lang IFn)
           (uncomplicate.neanderthal.internal.host MKL)))

(with-release [x (dv 0.3 0.9)
               w1 (dge 4 2 [0.3 0.6
                            0.1 2
                            0.9 3.7
                            0.0 1.0]
                    {:layout :row})
               h1 (dv 4)
               w2 (dge 1 4 [0.75 0.15 0.22 0.33])
               y (dv 1)]
  (mv! w1 x h1)
  (mv! w2 h1 y)
  y)

;(defn part-1
;  [x ws wf]
;  (with-release [h (dv (mrows (first ws)))] ; oversimplification regarding its size
;    (reduce (fn [acc w]
;              (with-release [c (copy acc)]
;                (mv! w c h)))
;      x ws)
;    (let [y (dv 1)]
;      (mv! wf h y))))
;
;(with-release [x (dv 0.3 0.9)
;               w1 (dge 4 2 [0.3 0.6
;                            0.1 2
;                            0.9 3.7
;                            0.0 1.0]
;                    {:layout :row})
;               w2 (dge 4 4 [1 0 0 0
;                            0 1 0 0
;                            0 0 1 0
;                            0 0 0 1]
;                    {:layout :row})
;               wf (dge 1 4 [0.75 0.15 0.22 0.33])
;               y (part-1 x [w1 w2 w2 w2 w2 w2 w2 w2 w2 w2 w2 w2 w2 w2 w2] wf)]
;  (println y))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(defn step! [threshold x]
  (fmap! signum (axpy! -1 threshold (fmax! threshold x x))))

(let [threshold (dv [1 2 3])
      x (dv [0 2 7])]
  (step! threshold x))

(def x (dv 0.3 0.9))
(def w1 (dge 4 2 [0.3 0.6
                  0.1 2
                  0.9 3.7
                  0.0 1.0]
          {:layout :row}))
(def threshold (dv 0.7 0.2 1.1 2))
(step! threshold (mv w1 x))

(def bias' (dv 0.7 0.2 1.1 2))
(def zero' (dv 4))
(step! zero' (axpy! -1.0 bias' (mv w1 x)))
(step! bias' (mv w1 x))

(defn relu! [threshold x]
  (axpy! -1.0 threshold (fmax! threshold x x)))

(let [threshold (dv [1 2 3])
      x (dv [0 2 7])]
  (prn (relu! threshold x)))
(relu! bias' (mv w1 x))

(tanh! (axpy! -1.0 bias' (mv w1 x)))
(def e (bigdec 2.71828))
(defn exp' [n] (loop [r e, n n] (if (= n 1) r (recur (* r e) (dec n)))))
(defn sigmoid [x] (.divide (exp' x) (inc (exp' x)) RoundingMode/HALF_EVEN))
(defn sigmoid! [x]
  (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))
(sigmoid! (axpy! -1.0 bias' (mv w1 x)))

(with-release [x (dv 0.3 0.9)
               w1 (dge 4 2 [0.3 0.6
                            0.1 2
                            0.9 3.7
                            0.0 1.0]
                    {:layout :row})
               bias1 (dv 0.7 0.2 1.1 2)
               h1 (dv 4)
               w2 (dge 1 4 [0.75 0.15 0.22 0.33])
               bias2 (dv 0.3)
               y (dv 1)]
  (tanh! (axpy! -1.0 bias1 (mv! w1 x h1)))
  (sigmoid! (axpy! -1.0 bias2 (mv! w2 h1 y)))
  y)

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(defprotocol Parameters
  (weights [this])
  (bias [this]))

(deftype FullyConnectedInference [w b h activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b)
    (release h))
  Parameters
  (weights [_] w)
  (bias [_] b)
  IFn
  (invoke [_ x]
    (activ-fn b (mv! w x h))))

(defn fully-connected [activ-fn in-dim out-dim]
  (let-release [w (dge out-dim in-dim)
                bias (dv out-dim)
                h (dv out-dim)]
    (->FullyConnectedInference w bias h activ-fn)))

(defn activ-sigmoid! [bias x]
  (axpy! -1.0 bias x)
  (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))

(defn activ-tanh! [bias x]
  (axpy! -1.0 bias x)
  (tanh! x))

(with-release [x (dv 0.3 0.9)
               layer-1 (fully-connected activ-sigmoid! 2 4)]
  (transfer! [0.3 0.1 0.0 #_0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
  (transfer! [0.7 0.2 1.1 2] (bias layer-1))
  (layer-1 x))

(with-release [x (dv 0.3 0.9)
               layer-1 (fully-connected activ-tanh! 2 4)
               layer-2 (fully-connected activ-sigmoid! 4 1)]
  (transfer! [0.3 0.1 0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
  (transfer! [0.7 0.2 1.1 2] (bias layer-1))
  (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
  (transfer! [0.3] (bias layer-2))
  (-> x layer-1 layer-2))

#_(with-release [x (dv 10000)
                 layer-1 (fully-connected activ-tanh! 10000 5000)
                 layer-2 (fully-connected activ-sigmoid! 5000 1000)
                 layer-3 (fully-connected activ-sigmoid! 1000 10)]
    (quick-bench (-> x layer-1 layer-2 layer-3)))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(let [t (dge 1000 1000)
      a (dge 1000 10000)
      y (dv 1000)
      b (dge 1000 10000)]
  #_(time (dotimes [i 10000]
            (mv! t (col a i) y)))
  (time (mm! 1.0 t a b)))

(let [x (dge 2 1 [0.3 0.9])
      w1 (dge 4 2 [0.3 0.6
                   0.1 2.0
                   0.9 3.7
                   0.0 1.0]
           {:layout :row})
      bias-matrix (dge 4 1 [0.7 0.2 1.1 2])]
  (mm w1 x)
  #_(sigmoid! (mm w1 x))
  (sigmoid! (axpy! -1.0 bias-matrix (mm w1 x))))

;https://www.mathsisfun.com/algebra/matrix-multiplying.html
;(mm
;  (dge 2 3 [1 2 3 4 5 6] {:layout :row})
;  (dge 3 2 [7 8 9 10 11 12] {:layout :row}))

(let-release [a (dge 3 2 (repeat 6 1000))]
  (with-release [x (dv 1 2 3)
                 y (dv 20 30)]
    (rk! 2 x y a)))

(let-release [a (dge 3 10 (repeat 30 1000))]
  (with-release [x (dv 1 2 3)
                 ones (entry! (dv 10) 1)]
    (rk! x ones a)))

(deftype FullyConnectedInference2 [w b h activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b)
    (release h))
  Parameters
  (weights [_] w)
  (bias [_] b)
  IFn
  (invoke [_ x]
    (activ-fn (axpy! -1.0 b (mv! w x h))))
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a)))))

(defn fully-connected2 [activ-fn in-dim out-dim]
  (let-release [w (dge out-dim in-dim)
                bias (dv out-dim)
                h (dv out-dim)]
    (->FullyConnectedInference2 w bias h activ-fn)))

(let-release [a (dge 4 1)]
  (with-release [x (dge 2 1 [0.3 0.9])
                 layer-1 (fully-connected2 sigmoid! 2 4)
                 ones (dv [1])]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (layer-1 x ones a)))

#_(with-release [x (dge 10000 10000)
                 ones (entry! (dv 10000) 1)
                 layer-1 (fully-connected2 tanh! 10000 5000)
                 a1 (dge 5000 10000)
                 layer-2 (fully-connected2 sigmoid! 5000 1000)
                 a2 (dge 1000 10000)
                 layer-3 (fully-connected2 sigmoid! 1000 10)
                 a3 (dge 10 10000)]
    (time (-> x (layer-1 ones a1) (layer-2 ones a2) (layer-3 ones a3))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(deftype FullyConnectedInference3 [w b activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b))
  Parameters
  (weights [_] w)
  (bias [_] b)
  IFn
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a)))))

(defn fully-connected3 [activ-fn in-dim out-dim]
  (let-release [w (dge out-dim in-dim)
                bias (dv out-dim)]
    (->FullyConnectedInference3 w bias activ-fn)))

(let-release [temp-a (dv 8)]
  (with-release [x (dge 2 2 [0.3 0.9 0.3 0.9])
                 ones (dv 1 1)
                 layer-1 (fully-connected3 tanh! 2 4)
                 a-1 (view-ge temp-a 4 2)
                 layer-2 (fully-connected3 sigmoid! 4 1)
                 a-2 (view-ge temp-a 1 2)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (layer-2 (layer-1 x ones a-1) ones a-2)))

(let-release [temp-odd (dv 8)
              temp-even (dv 2)]
  (with-release [x (dge 2 2 [0.3 0.9 0.3 0.9])
                 ones (dv 1 1)
                 layer-1 (fully-connected3 tanh! 2 4)
                 a-1 (view-ge temp-odd 4 2)
                 layer-2 (fully-connected3 sigmoid! 4 1)
                 a-2 (view-ge temp-even 1 2)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (layer-2 (layer-1 x ones a-1) ones a-2)))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(defn fully-connected4 [factory activ-fn in-dim out-dim]
  (let-release [w (ge factory out-dim in-dim)
                bias (vctr factory out-dim)]
    (->FullyConnectedInference3 w bias activ-fn)))

(defn this-particular-network [factory]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones (vctr factory 1 1)
                 layer-1 (fully-connected4 factory tanh! 2 4)
                 a-1 (ge factory 4 2)
                 layer-2 (fully-connected4 factory sigmoid! 4 1)
                 a-2 (ge factory 1 2)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (transfer (layer-2 (layer-1 x ones a-1) ones a-2))))

#_(this-particular-network native-double)

#_(opencl/with-default
    (with-release [opencl-factory (opencl-float *context* *command-queue*)]
      (this-particular-network opencl-factory)))

#_(opencl/with-default
    (with-release [factory (opencl-float *context* *command-queue*)
                   x (ge factory 10000 10000)
                   ones (entry! (vctr factory 10000) 1)
                   layer-1 (fully-connected4 factory tanh! 10000 5000)
                   a1 (ge factory 5000 10000)
                   layer-2 (fully-connected4 factory sigmoid! 5000 1000)
                   a2 (ge factory 1000 10000)
                   layer-3 (fully-connected4 factory sigmoid! 1000 10)
                   a3 (ge factory 10 10000)]
      (layer-1 x ones a1) ; warm up
      (finish!)
      (time (do (-> x (layer-1 ones a1) (layer-2 ones a2) (layer-3 ones a3))
                (finish!)))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(with-release [x (dv 1 2 3)
               y (dv 4 5 6)]
  (mul! x y)
  (transfer x))

(with-release [a (dge 2 3 [1 2 3 4 5 6])
               b (dge 2 3 [7 8 9 10 11 12])]
  (mul! a b)
  (transfer a))

(let [x (dv 0 0.5 0.7 1)]
  (exp x)
  (sin x)
  (cos x))
#_(defn sigmoid! [x]
    (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))

(defn sigmoid-prim!
  ([x!]
   (let [x-raw! (raw x!)]
     (sigmoid-prim! x! x-raw!)))
  ([x! prim!]
   (sigmoid! x!)
   (mul! (linear-frac! -1.0 x! 1.0 prim!) x!)))

(let [x (dv 0.1 0.5 0.9 (/ Math/PI 2.0))]
  [(transfer x)
   (sigmoid! (transfer x))
   (sigmoid-prim! x)])

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(defprotocol ActivationProvider
  (activation-fn [this]))

(deftype FullyConnectedInference5 [w b activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b))
  Parameters
  (weights [_] w)
  (bias [_] b)
  ActivationProvider
  (activation-fn [_] activ-fn)
  IFn
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a))))) ; Not sure why it is -1.0 instead of 1.0

(defn fully-connected5 [factory activ-fn in-dim out-dim]
  (let-release [w (ge factory out-dim in-dim)
                bias (vctr factory out-dim)]
    (->FullyConnectedInference5 w bias activ-fn)))

(defprotocol Backprop
  (forward [this])
  (backward [this]))

(defprotocol Transfer
  (input [this])
  (output [this])
  (ones [this]))

(deftype FullyConnectedTraining [w b a-1 z a ones-vctr activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones-vctr))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop
  (forward [_]
    (activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward [_]
    (throw (ex-info "TODO" nil))))

(defn sigmoid!
  ([x]
   (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))
  ([x y]
   (linear-frac! 0.5 (tanh! (scal! 0.5 (copy! x y))) 0.5)))

(defn training-layer
  ([inference-layer input ones-vctr]
   (let-release [w (view (weights inference-layer))
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining w b a-1 z a o (activation-fn inference-layer))))
  ([inference-layer previous-backdrop]
   (training-layer inference-layer
     (output previous-backdrop) (ones previous-backdrop))))

(defn this-particular-network-2 [factory]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones (vctr factory 1 1)
                 layer-1 (fully-connected5 factory tanh! 2 4)
                 layer-2 (fully-connected5 factory sigmoid! 4 1)
                 training-layer-1 (training-layer layer-1 x ones)
                 training-layer-2 (training-layer layer-2 training-layer-1)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (forward training-layer-1)
    (forward training-layer-2)
    (transfer (output training-layer-2))))

(this-particular-network-2 native-float)

#_(opencl/with-default
    (with-release [opencl-factory (opencl-float *context* *command-queue*)]
      (this-particular-network-2 opencl-factory)))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(defn sigmoid!
  ([]
   sigmoid-prim!)
  ([x]
   (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))
  ([x y]
   (linear-frac! 0.5 (tanh! (scal! 0.5 (copy! x y))) 0.5)))

(defprotocol Activation
  (activ [_ z a!])
  (prime [_ z]))

(deftype SigmoidActivation [work]
  Releaseable
  (release [_]
    (release work))
  Activation
  (activ [_ z a!]
    (linear-frac! 0.5 (tanh! (scal! 0.5 (copy! z a!))) 0.5))
  (prime [_ z!]
    (linear-frac! 0.5 (tanh! (scal! 0.5 z!)) 0.5)
    (mul! z! (linear-frac! -1.0 z! 1.0 work))))

(defn sigmoid
  ([]
   (fn [z]
     (let-release [work (raw z)]
       (->SigmoidActivation work))))
  ([z!]
   (linear-frac! 0.5 (tanh! (scal! 0.5 z!)) 0.5)))

(let [z (fge 2 3 [-0.6 0 0.2 0.5 0.7 1])
      a (raw z)]
  (with-release [activation ((sigmoid) z)]
    {:function (activ activation z a)
     :derivative (prime activation z)}))

(deftype TanhActivation []
  Activation
  (activ [_ z a!]
    (tanh! z a!))
  (prime [_ z]
    (sqr! (inv! (cosh! z)))))

(defn tanh
  ([]
   (fn [_]
     (->TanhActivation)))
  ([z!]
   (tanh! z!)))

(deftype FullyConnectedTraining2 [w b a-1 z a ones-vctr activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones-vctr)
    (release activ-fn))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop
  (forward [_]
    (activ activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward [_]
    (mul! (prime activ-fn z) a) ; z updated
    "TODO"))

(defn training-layer2
  ([inference-layer input ones-vctr]
   (let-release [w (view (weights inference-layer))
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining2 w b a-1 z a o (((activation-fn inference-layer)) z))))
  ([inference-layer previous-backprop]
   (training-layer2 inference-layer
     (output previous-backprop) (ones previous-backprop))))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones (vctr factory 1 1)
                 layer-1 (fully-connected5 factory tanh 2 4)
                 layer-2 (fully-connected5 factory sigmoid 4 1)
                 training-layer-1 (training-layer2 layer-1 x ones)
                 training-layer-2 (training-layer2 layer-2 training-layer-1)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (forward training-layer-1)
    (forward training-layer-2)
    (backward training-layer-2)
    (backward training-layer-1)
    (transfer (output training-layer-2))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

; forward - input a-1, output a

(defprotocol Backprop2
  (forward2 [this])
  (backward2 [this eta]))

(deftype FullyConnectedTraining3 [v w b a-1 z a ones-vctr activ-fn]
  Releaseable
  (release [_]
    (release v)
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones-vctr)
    (release activ-fn))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop2
  (forward2 [_]
    (activ activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward2 [_ eta]
    (let [eta-avg (/ (- (double eta)) (dim ones-vctr))]
      (mul! (prime activ-fn z) a) ; z is now d. updating layers "in advance" so a is w^l+1T x d^l+1 (was put in here by l+1 layer).
      (mm! eta-avg z (trans a-1) 0.0 v) ; weights delta is now in v
      (mm! 1.0 (trans w) z 0.0 a-1) ; see the comment two lines above (this is the l+1 layer)
      (mv! eta-avg z ones-vctr 1.0 b) ; bias has been updated
      (axpy! 1.0 v w) ; weights have been updated
      )))

(defn training-layer3
  ([inference-layer input ones-vctr]
   (let-release [w (view (weights inference-layer))
                 v (raw w)
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining3 v w b a-1 z a o (((activation-fn inference-layer)) z))))
  ([inference-layer previous-backprop]
   (training-layer3 inference-layer
     (output previous-backprop) (ones previous-backprop))))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones (vctr factory 1 1)
                 layer-1 (fully-connected5 factory tanh 2 4)
                 layer-2 (fully-connected5 factory sigmoid 4 1)
                 training-layer-1 (training-layer3 layer-1 x ones)
                 training-layer-2 (training-layer3 layer-2 training-layer-1)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (forward2 training-layer-1)
    (forward2 training-layer-2)
    (backward2 training-layer-2 0.05)
    (backward2 training-layer-1 0.05)
    (transfer (output training-layer-2))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(comment
  (def input (fv 1000))
  (def output (fv 8))
  (def inference (net factory 1000
                   [(fully-connected 256 sigmoid)
                    (fully-connected 64 tanh)
                    (fully-connected 16 sigmoid)]))
  (def train (training inference input output quadratic-cost))
  (sgd train 20))

(deftype NeuralNetworkInference [layers
                                 ^long max-width-1
                                 ^long max-width-2]
  Releaseable
  (release [_]
    (doseq [l layers]
      (release l)))
  IFn
  (invoke [_ x ones-vctr temp-1! temp-2!]
    (let [batch (dim ones-vctr)]
      (loop [x x
             v1 temp-1!
             v2 temp-2!
             layers layers]
        (if layers
          (recur
            (let [layer (first layers)]
              (layer x ones-vctr (view-ge v1 (mrows (weights layer)) batch)))
            v2 v1 (next layers))
          x))))
  (invoke [this x a!]
    (let [cnt (count layers)]
      (if (zero? cnt)
        (copy! x a!)
        (with-release [ones-vctr (entry! (vctr x (ncols x)) 1.0)]
          (if (= 1 cnt)
            ((layers 0) x ones-vctr a!)
            (with-release [temp-1 (vctr x (* max-width-1 (dim ones-vctr)))]
              (if (= 2 cnt)
                (this x ones-vctr temp-1 a!)
                (with-release [temp-2 (vctr x (* max-width-2 (dim ones-vctr)))]
                  (copy! (this x ones-vctr temp-1 temp-2) a!)))))))))
  (invoke [this x]
    (let-release [a (ge x (mrows (weights (peek layers))) (ncols x))]
      (this x a))))

(defn inference-network [factory in-dim layers]
  (let [out-sizes (map #(%) layers)
        in-sizes (cons in-dim out-sizes)
        max-width-1 (apply max (take-nth 2 out-sizes))
        max-width-2 (apply max (take-nth 2 (rest out-sizes)))]
    (let-release [layers (mapv (fn [layer-fn in-size]
                                 (layer-fn factory in-size))
                           layers
                           in-sizes)]
      (->NeuralNetworkInference layers max-width-1 max-width-2))))

(defn fully-connected7
  ([factory in-dim out-dim activ]
   (let-release [w (ge factory out-dim in-dim)
                 bias (vctr factory out-dim)]
     (->FullyConnectedInference5 w bias activ)))
  ([out-dim activ]
   (fn
     ([factory in-dim]
      (fully-connected7 factory in-dim out-dim activ))
     ([]
      out-dim))))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 a (ge factory 1 2)
                 inference (inference-network factory 2
                             [(fully-connected7 4 tanh)
                              (fully-connected7 1 sigmoid)])
                 layers (.layers inference)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights (layers 0)))
    (transfer! [0.7 0.2 1.1 2] (bias (layers 0)))
    (transfer! [0.75 0.15 0.22 0.33] (weights (layers 1)))
    (transfer! [0.3] (bias (layers 1)))
    (transfer (inference x a))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(comment
  (def input (fv 1000))
  (def output (fv 8))
  (def inference (inference-network factory 1000
                   [(fully-connected 256 sigmoid)
                    (fully-connected 64 tanh)
                    (fully-connected 16 sigmoid)]))
  (def train (training inference input))
  (sgd train output quadratic-cost 20 0.05))

(defprotocol NeuralNetwork
  (layers [this]))

(deftype NeuralNetworkTraining [forward-layers backward-layers]
  Releaseable
  (release [_]
    (doseq [layer forward-layers]
      (release layer)))
  NeuralNetwork
  (layers [_]
    forward-layers)
  Transfer
  (input [_]
    (input (first forward-layers)))
  (output [_]
    (output (first backward-layers)))
  (ones [_]
    (ones (first backward-layers)))
  Backprop2
  (forward2 [_]
    (doseq [layer forward-layers]
      (forward2 layer))
    (output (first backward-layers)))
  (backward2 [_ eta]
    (doseq [layer backward-layers]
      (backward2 layer eta))))

(defn training-network
  [inference input]
  (let [layers (.layers inference)]
    (let-release [ones-vctr (entry! (raw (row input 0)) 1.0)
                  backward-layers
                  (reduce (fn [acc layer]
                            (cons (training-layer3 layer (first acc)) acc))
                    (list (training-layer3 (first layers) input ones-vctr))
                    (rest layers))]
      (->NeuralNetworkTraining (reverse backward-layers) backward-layers))))

(defn sgd
  [network out cost! epochs eta]
  (dotimes [n epochs]
    (forward2 network)
    (cost! out (output network))
    (backward2 network eta))
  (cost! (output network)))

(defn quadratic-cost!
  ([y-a]
   (/ (sqr (nrm2 y-a)) (* 2 (dim y-a))))
  ([y a!]
   (axpy! -1.0 y a!)))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 y (ge factory 1 2 [0.5 0.5])
                 inference (inference-network factory 2
                             [(fully-connected7 4 tanh)
                              (fully-connected7 1 sigmoid)])
                 inf-layers (.layers inference)
                 training (training-network inference x)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights (inf-layers 0)))
    (transfer! [0.7 0.2 1.1 2] (bias (inf-layers 0)))
    (transfer! [0.75 0.15 0.22 0.33] (weights (inf-layers 1)))
    (transfer! [0.3] (bias (inf-layers 1)))
    {:untrained (transfer (inference x))
     :cost [(sgd training y quadratic-cost! 1 0.05)
            (sgd training y quadratic-cost! 20 0.05)
            (sgd training y quadratic-cost! 200 0.05)
            (sgd training y quadratic-cost! 2000 0.05)]
     :trained (transfer (inference x))
     :messed-up-inputs (transfer x)}))

(deftype FullyConnectedTraining4 [v w b a-1 z a ones-vctr activ-fn first?]
  Releaseable
  (release [_]
    (release v)
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones-vctr)
    (release activ-fn))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop2
  (forward2 [_]
    (activ activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward2 [_ eta]
    (let [eta-avg (/ (- (double eta)) (dim ones-vctr))]
      (mul! (prime activ-fn z) a) ; z is now d. updating layers "in advance" so a is w^l+1T x d^l+1 (was put in here by l+1 layer).
      (mm! eta-avg z (trans a-1) 0.0 v) ; weights delta is now in v
      (when-not first? (mm! 1.0 (trans w) z 0.0 a-1)) ; see the comment two lines above (this is the l+1 layer)
      (mv! eta-avg z ones-vctr 1.0 b) ; bias has been updated
      (axpy! 1.0 v w) ; weights have been updated
      )))

(defn training-layer4
  ([inference-layer input ones-vctr first?]
   (let-release [w (view (weights inference-layer))
                 v (raw w)
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining4 v w b a-1 z a o
       (((activation-fn inference-layer)) z) first?)))
  ([inference-layer input ones-vctr]
   (training-layer4 inference-layer input ones-vctr true))
  ([inference-layer previous-backprop]
   (training-layer4 inference-layer
     (output previous-backprop) (ones previous-backprop) false)))

(defn training-network2
  [inference input]
  (let [layers (.layers inference)]
    (let-release [ones-vctr (entry! (raw (row input 0)) 1.0)
                  backward-layers
                  (reduce (fn [acc layer]
                            (cons (training-layer4 layer (first acc)) acc))
                    (list (training-layer4 (first layers) input ones-vctr))
                    (rest layers))]
      (->NeuralNetworkTraining (reverse backward-layers) backward-layers))))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 y (ge factory 1 2 [0.5 0.5])
                 inference (inference-network factory 2
                             [(fully-connected7 4 tanh)
                              (fully-connected7 1 sigmoid)])
                 inf-layers (.layers inference)
                 training (training-network2 inference x)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights (inf-layers 0)))
    (transfer! [0.7 0.2 1.1 2] (bias (inf-layers 0)))
    (transfer! [0.75 0.15 0.22 0.33] (weights (inf-layers 1)))
    (transfer! [0.3] (bias (inf-layers 1)))
    {:untrained (transfer (inference x))
     :cost [(sgd training y quadratic-cost! 1 0.05)
            (sgd training y quadratic-cost! 20 0.05)
            (sgd training y quadratic-cost! 200 0.05)
            (sgd training y quadratic-cost! 2000 0.05)]
     :trained (transfer (inference x))
     :inputs-are-unchanged (transfer x)}))

(defn sgd2
  ([network out cost! epochs eta]
   (dotimes [n epochs]
     (forward2 network)
     (cost! out (output network))
     (backward2 network eta))
   (cost! (output network)))
  ([network out cost! options]
   (mapv (fn [[epochs eta]] (sgd2 network out cost! epochs eta)) options)))

#_(let [factory native-float]
    (with-release [x (ge factory 2000 4000)
                   y (entry! (ge factory 10 4000) 0.33)
                   inference (inference-network factory 2000
                               [(fully-connected7 5000 tanh)
                                (fully-connected7 1000 sigmoid)
                                (fully-connected7 10 sigmoid)])
                   training (training-network2 inference x)]
      (time (sgd2 training y quadratic-cost! (repeat 10 [1 0.05])))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 y (ge factory 1 2 [0.5 0.5])
                 inference (inference-network factory 2
                             [(fully-connected7 4 tanh)
                              (fully-connected7 1 sigmoid)])
                 inf-layers (.layers inference)
                 training (training-network2 inference x)]
    (transfer! [3 1 9 0 6 20 37 10] (weights (inf-layers 0)))
    (transfer! [7 2 11 2] (bias (inf-layers 0)))
    (transfer! [75 15 22 33] (weights (inf-layers 1)))
    (transfer! [3] (bias (inf-layers 1)))
    (sgd2 training y quadratic-cost! 2000 0.05)
    (transfer (inference x))))

(fmap! (fn [_] (rand)) (fge 3 2))

(defn rand-uniform ^double [^double _]
  (double (rand)))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 y (ge factory 1 2 [0.5 0.5])
                 inference (inference-network factory 2
                             [(fully-connected7 4 tanh)
                              (fully-connected7 1 sigmoid)])
                 inf-layers (.layers inference)
                 training (training-network2 inference x)]
    (fmap! rand-uniform (weights (inf-layers 0)))
    (fmap! rand-uniform (bias (inf-layers 0)))
    (fmap! rand-uniform (weights (inf-layers 1)))
    (fmap! rand-uniform (bias (inf-layers 1)))
    (sgd2 training y quadratic-cost! 2000 0.05)
    (transfer (inference x))))

(defn rand-gaussian ^double [^double _]
  (let [u1 (rand-uniform Double/NaN)
        u2 (rand-uniform Double/NaN)]
    (double (* (sqrt (* -2.0 (log u1))) (Math/sin (* 2.0 pi u2))))))

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 y (ge factory 1 2 [0.5 0.5])
                 inference (inference-network factory 2
                             [(fully-connected7 4 tanh)
                              (fully-connected7 1 sigmoid)])
                 inf-layers (.layers inference)
                 training (training-network2 inference x)]
    (fmap! rand-gaussian (weights (inf-layers 0)))
    (fmap! rand-gaussian (bias (inf-layers 0)))
    (fmap! rand-gaussian (weights (inf-layers 1)))
    (fmap! rand-gaussian (bias (inf-layers 1)))
    (sgd2 training y quadratic-cost! 2000 0.05)
    (transfer (inference x))))

(defn init-layer! [layer!]
  (let [w (weights layer!)]
    (scal! (sqrt (/ 2.0 (+ (mrows w) (ncols w))))
      (fmap! rand-gaussian w))
    (fmap! rand-gaussian (bias layer!))
    layer!))

(defn init! [network!]
  (doseq [layer (.layers network!)]
    (init-layer! layer))
  network!)

(let [factory native-float]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 y (ge factory 1 2 [0.5 0.5])
                 inference (init! (inference-network factory 2
                                    [(fully-connected7 4 tanh)
                                     (fully-connected7 1 sigmoid)]))
                 training (training-network2 inference x)]
    (sgd2 training y quadratic-cost! 2000 0.05)
    (transfer (inference x))))

#_(with-release [x (ge native-double 100 100)]
    (quick-bench (fmap! rand-uniform x)))

#_(with-release [x (ge native-double 100 100)]
    (quick-bench (fmap! rand-gaussian x)))

(defn init-weights! [w b]
  (scal! (/ 1.0 (ncols w)) (fmap! rand-gaussian w))
  (fmap! rand-gaussian b))

(defn init-layer2! [layer!]
  (let [w (weights layer!)
        b (bias layer!)]
    (if (= :cpu (device w))
      (init-weights! w b)
      (with-release [native-w (ge (native-factory w) (mrows w) (ncols w))
                     native-b (vctr (native-factory b) (dim b))]
        (init-weights! native-w native-b)
        (transfer! native-w w)
        (transfer! native-b b)))
    layer!))

(defn init2! [network!]
  (doseq [layer (.layers network!)]
    (init-layer2! layer))
  network!)

#_(let [factory native-float]
    (with-release [x (ge factory 10000 100)
                   y (entry! (ge factory 10 100) 0.33)
                   inference (time (init2! (inference-network factory 10000
                                             [(fully-connected7 5000 tanh)
                                              (fully-connected7 1000 sigmoid)
                                              (fully-connected7 10 sigmoid)])))
                   training (training-network2 inference x)]
      (time (sgd2 training y quadratic-cost! 10 0.05))))

;(let-release [stream (direct-buffer Long/BYTES)]
;  (let [seed (rand-int 1000)
;        err (MKL/vslNewStreamARS5 seed stream)]
;    (if (= 0 err)
;      (defn float-gaussian-sample! [res]
;        (let [err (MKL/vsRngGaussian stream (dim res) (buffer res)
;                    (* (mrows res) (ncols res)) 0 1 ; guessing here
;                    )]
;          (if (= 0 err)
;            res
;            (throw (ex-info "MKL error" nil)))))
;      (ex-info "MKL error" nil))))
;
;(defn init-weights3! [w b]
;  (scal! (/ 1.0 (ncols w)) (float-gaussian-sample! w))
;  (float-gaussian-sample! b))
;
;(defn init-layer3! [layer!]
;  (let [w (weights layer!)
;        b (bias layer!)]
;    (if (= :cpu (device w))
;      (init-weights3! w b)
;      (with-release [host-w (ge (native-factory w) (mrows w) (ncols w))
;                     host-b (vctr (native-factory b) (dim b))]
;        (init-weights3! host-w host-b)
;        (transfer! host-w w)
;        (transfer! host-b b)))
;    layer!))
;
;(defn init3! [network!]
;  (doseq [layer (.layers network!)]
;    (init-layer3! layer))
;  network!)
;
;(let [factory native-float]
;  (with-release [x (ge factory 10000 100)
;                 y (entry! (ge factory 10 100) 0.33)
;                 inference (time (init3! (inference-network factory 10000
;                                           [(fully-connected6 5000 tanh)
;                                            (fully-connected6 1000 sigmoid)
;                                            (fully-connected6 10 sigmoid)])))
;                 training (training-network2 inference x)]
;    (time (sgd2 training y quadratic-cost! 10 0.05)))) ; crashes JVM

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(defn my-fn ^double [xs]
  (+ (math/sin (real/entry xs 0))
    (math/cos (real/entry xs 1))
    (math/tanh (real/entry xs 2))
    (math/sqr (real/entry xs 3))))

(my-fn (vctr native-float [0.3 0.1 0.9 0.33]))

(def x-train (fmap! rand-uniform (fge 4 10000)))

(def y-train (fge 1 10000 (map my-fn (cols x-train))))

(def x-test (fmap! rand-uniform (fge 4 5)))

(def y-test (fge 1 5 (map my-fn (cols x-test))))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 16 sigmoid)
                             (fully-connected7 64 tanh)
                             (fully-connected7 8 tanh)
                             (fully-connected7 1 sigmoid)]))
        training (training-network2 inference x-train)]
    #_(weights (last (.layers training)))
    (sgd2 training y-train quadratic-cost! [[1 0.05] [1000 0.03] [100 0.01]])
    (inference x-test))

(deftype LinearActivation []
  Activation
  (activ [_ z a!]
    (copy! z a!))
  (prime [_ z!]
    (entry! z! 1)))

(defn linear
  ([]
   (fn [_]
     (->LinearActivation)))
  ([z!]
   z!))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 16 sigmoid)
                             (fully-connected7 64 tanh)
                             (fully-connected7 8 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network2 inference x-train)]
    #_(weights (last (.layers training)))
    #_(sgd2 training y-train quadratic-cost! [[1 0.05] [1000 0.03] [100 0.01]])
    #_(inference x-test)
    #_[(inference x-test)
       (sgd2 training y-train quadratic-cost! 1 0.05)
       (inference x-test)
       (sgd2 training y-train quadratic-cost! 1 0.05)
       (inference x-test)
       (sgd2 training y-train quadratic-cost! 10 0.05)
       (inference x-test)
       (sgd2 training y-train quadratic-cost! 100 0.05)
       (inference x-test)
       (sgd2 training y-train quadratic-cost! 100 0.03)
       (inference x-test)
       (sgd2 training y-train quadratic-cost! 1000 0.01)
       (inference x-test)
       (sgd2 training y-train quadratic-cost! [[100 0.03] [100 0.01] [100 0.005]
                                               [100 0.001]])
       (inference x-test)]
    [(time (sgd2 training y-train quadratic-cost! 40000 0.05))
     (inference x-test)
     (axpy! -1 y-test (inference x-test))])

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 8 sigmoid)
                             (fully-connected7 16 tanh)
                             (fully-connected7 4 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network2 inference x-train)]
    #_(time (sgd2 training y-train quadratic-cost! 4000 0.05))
    (time (sgd2 training y-train quadratic-cost! 40000 0.05)))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 8 sigmoid)
                             (fully-connected7 4 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network2 inference x-train)]
    (time (sgd2 training y-train quadratic-cost! 4000 0.05)))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 4 sigmoid)
                             (fully-connected7 8 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network2 inference x-train)]
    (time (sgd2 training y-train quadratic-cost! 4000 0.05)))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 8 sigmoid)
                             (fully-connected7 8 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network2 inference x-train)]
    [(time (sgd2 training y-train quadratic-cost! 4000 0.05))
     (inference x-test)
     (axpy! -1 y-test (inference x-test))])

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 32 sigmoid)
                             (fully-connected7 16 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network2 inference x-train)]
    [(sgd2 training y-train quadratic-cost! #_15 100 0.3)
     (weights (first (.layers inference)))
     (map amax (map weights (.layers inference)))])

(deftype FullyConnectedTraining5 [v w b a-1 z a ones-vctr activ-fn first?]
  Releaseable
  (release [_]
    (release v)
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones-vctr)
    (release activ-fn))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop2
  (forward2 [_]
    (activ activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward2 [_ [eta lambda]]
    (let [eta-avg (/ (- (double eta)) (dim ones-vctr))]
      (mul! (prime activ-fn z) a) ; z is now d. updating layers "in advance" so a is w^l+1T x d^l+1 (was put in here by l+1 layer).
      (mm! eta-avg z (trans a-1) 0.0 v) ; weights delta is now in v
      (when-not first? (mm! 1.0 (trans w) z 0.0 a-1)) ; see the comment two lines above (this is the l+1 layer)
      (mv! eta-avg z ones-vctr 1.0 b) ; bias has been updated
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w) ; weights have been updated (with L^2 Regularization)
      )))

(defn training-layer5
  ([inference-layer input ones-vctr first?]
   (let-release [w (view (weights inference-layer))
                 v (zero w)
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining5 v w b a-1 z a o
       (((activation-fn inference-layer)) z) first?)))
  ([inference-layer input ones-vctr]
   (training-layer5 inference-layer input ones-vctr true))
  ([inference-layer previous-backprop]
   (training-layer5 inference-layer
     (output previous-backprop) (ones previous-backprop) false)))

(defn training-network3
  [inference input]
  (let [layers (.layers inference)]
    (let-release [ones-vctr (entry! (raw (row input 0)) 1.0)
                  backward-layers
                  (reduce (fn [acc layer]
                            (cons (training-layer5 layer (first acc)) acc))
                    (list (training-layer5 (first layers) input ones-vctr))
                    (rest layers))]
      (->NeuralNetworkTraining (reverse backward-layers) backward-layers))))

(def x-test2 (fmap! rand-uniform (fge 4 10000)))

(def y-test2 (fge 1 10000 (map my-fn (cols x-test2))))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 32 sigmoid)
                             (fully-connected7 16 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network3 inference x-train)]
    [(sgd2 training y-train quadratic-cost! 15 [0.3 0.9])
     (map amax (map weights (.layers inference)))
     (sgd2 training y-train quadratic-cost! 150 [0.3 0.9])
     (map amax (map weights (.layers inference)))
     (sgd2 training y-train quadratic-cost! 1000 [0.3 0.9])
     (map amax (map weights (.layers inference)))
     (sgd2 training y-train quadratic-cost! 1000 [0.3 0.9])
     (map amax (map weights (.layers inference)))
     (inference x-test2)
     y-test2
     (quadratic-cost! (axpy! -1.0 y-test2 (inference x-test2)))])

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(deftype FullyConnectedTraining6 [v w b a-1 z a ones-vctr activ-fn first?]
  Releaseable
  (release [_]
    (release v)
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones-vctr)
    (release activ-fn))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop2
  (forward2 [_]
    (activ activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward2 [_ [eta lambda mu]]
    (let [eta-avg (/ (- (double eta)) (dim ones-vctr))]
      (mul! (prime activ-fn z) a) ; z is now d. updating layers "in advance" so a is w^l+1T x d^l+1 (was put in here by l+1 layer).
      (mm! eta-avg z (trans a-1) mu v) ; weights delta is now in v
      (when-not first? (mm! 1.0 (trans w) z 0.0 a-1)) ; see the comment two lines above (this is the l+1 layer)
      (mv! eta-avg z ones-vctr 1.0 b) ; bias has been updated
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w) ; weights have been updated (with L^2 Regularization)
      )))

(defn training-layer6
  ([inference-layer input ones-vctr first?]
   (let-release [w (view (weights inference-layer))
                 v (zero w)
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining6 v w b a-1 z a o
       (((activation-fn inference-layer)) z) first?)))
  ([inference-layer input ones-vctr]
   (training-layer6 inference-layer input ones-vctr true))
  ([inference-layer previous-backprop]
   (training-layer6 inference-layer
     (output previous-backprop) (ones previous-backprop) false)))

(defn training-network4
  [inference input]
  (let [layers (.layers inference)]
    (let-release [ones-vctr (entry! (raw (row input 0)) 1.0)
                  backward-layers
                  (reduce (fn [acc layer]
                            (cons (training-layer6 layer (first acc)) acc))
                    (list (training-layer6 (first layers) input ones-vctr))
                    (rest layers))]
      (->NeuralNetworkTraining (reverse backward-layers) backward-layers))))

#_(let [inference (init2! (inference-network native-float 4
                            [(fully-connected7 32 sigmoid)
                             (fully-connected7 16 tanh)
                             (fully-connected7 1 linear)]))
        training (training-network4 inference x-train)]
    [(sgd2 training y-train quadratic-cost! 50 [0.05 0.1 0.1])
     (quadratic-cost! (axpy! -1.0 y-test2 (inference x-test2)))
     (sgd2 training y-train quadratic-cost! 1000 [0.05 0.1 #_0 0.1])
     (quadratic-cost! (axpy! -1.0 y-test2 (inference x-test2)))
     (sgd2 training y-train quadratic-cost! 4000 [0.01 0.01 0.1])])

#_(for [eta-lambda-mu [[0.2 0 0]
                       [0.2 0.02 0]
                       [0.2 0.02 0.01]]]
    (let [inference (init2! (inference-network native-float 4
                              [(fully-connected7 8 sigmoid)
                               (fully-connected7 8 tanh)
                               (fully-connected7 1 linear)]))
          training (training-network4 inference x-train)]
      (time (sgd2 training y-train quadratic-cost!
              (repeat #_3 8 [#_4000 50 eta-lambda-mu])))))

; *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

(let [a (fge 3 2)] (random/rand-normal! 33 2.5 a))

#_(with-release [x (fv 300000000), y (fv 300000000)]
    (time (do (random/rand-uniform! x)
              (random/rand-uniform! y)
              :done)))

(let [a (fge 4 3), sub-a (submatrix a 0 1 4 2)]
  (random/rand-uniform! 0 2 sub-a)
  a)
(random/rand-uniform! (random/rng-state native-float 11) 2 3 (fv 3))
