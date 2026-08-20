# Not Hotdog 🌭

A modern recreation of Tim Anglade's *Silicon Valley* app. Point a camera at something and it
tells you whether it is a hotdog. That's the whole product.

**[Try it →](https://austenstone.github.io/not-hotdog/)**

Everything runs on-device. No inference server, no API keys, no per-request cost — the same
constraint Anglade designed around in 2017, just with tooling that no longer requires
hand-written Objective-C++ bridges.

---

## The one decision that shaped everything

**A single quantized `.tflite` artifact runs on both web and mobile.**

| Target | Runtime | Package |
| --- | --- | --- |
| Web | LiteRT.js (WebGPU, WASM fallback) | [`@litertjs/core`](https://www.npmjs.com/package/@litertjs/core) |
| Mobile | LiteRT via JSI | [`react-native-fast-tflite`](https://www.npmjs.com/package/react-native-fast-tflite) |

The conventional path would ship a TensorFlow.js bundle to the browser and a `.tflite` to mobile —
two artifacts, two conversion steps, two chances to drift apart.

That path is also **impossible on Python 3.13**, which is worth writing down because the failure
is non-obvious:

> `tensorflowjs` depends on `tensorflow-decision-forests`, which publishes wheels for cp39–cp312
> only. There is no cp313 wheel, so `pip install tensorflowjs` hard-fails on any Python 3.13
> environment. The dependency is entirely incidental — the converter doesn't need decision
> forests to convert a MobileNet — but pip doesn't care.

LiteRT.js is Google's official successor to TFJS, runs the `.tflite` directly, and is faster
besides. So the blocker pushed us toward the better architecture. **Do not reintroduce
`@tensorflow/tfjs` as an inference dependency.**

Related: `tensorflow-metal` also caps at cp312 and breaks under Keras 3, so **there is no GPU
here**. That single fact drives the two-stage training design below.

---

## Layout

```
training/     uv-managed Python package — data curation through .tflite export
models/       committed .tflite + metadata.json (the contract both clients read)
web/          Vite + React + TypeScript, LiteRT.js
mobile/       React Native, vision-camera + fast-tflite
```

---

## Where the accuracy actually comes from

Not the architecture. MobileNetV2 transfer learning is a solved, boring choice. Accuracy is
almost entirely a function of **which negatives the model sees**.

Food-101 contains exactly 1,000 hotdog images and no amount of cleverness changes that. So all the
leverage is on the other side of the ratio:

| Group | Source | Train count | Why |
| --- | --- | --- | --- |
| Hotdog | `food101/hot_dog` | 4,500 | 750 images × (4 augmented + 2 screen variants) |
| Hard negative | 21 curated Food-101 classes | 15,750 | Kept **whole** — the lookalikes |
| Easy negative | remaining ~79 classes | 20,000 | Subsampled |
| Non-food | `imagenette` | 8,000 | Teaches "is this food at all" |

**The asymmetry is the oversampling.** Hard negatives are kept complete while easy negatives are
subsampled, which raises the lookalike proportion without duplicating a single image. Net ratio
lands near 10:1, against the original app's ~49:1.

The 21 hard negatives are chosen for the specific failure each would cause. `lobster_roll_sandwich`
and `breakfast_burrito` are the genuinely dangerous ones — a split bun and a browned cylinder are
exactly the shape the model looks for. Full list in
[`training/src/nothotdog/config.py`](training/src/nothotdog/config.py).

### The screen trick

Users photograph a hotdog on a monitor to cheat. The original app countered this by collecting
photos of LCD screens. We **synthesize** it instead — sinusoidal moiré, scanlines, a glare
gradient, and bezel crops applied programmatically. Cheaper, and infinitely more of it.

`SCREEN_LABEL` decides the policy: `hotdog` (default) treats a hotdog behind glass as a hotdog
(robustness), `not_hotdog` reproduces the show's anti-cheat spirit.

### Two leakage traps this pipeline is built to avoid

These are easy to accidentally undo, so they get called out here:

1. **Augmentation leakage.** If only positives were augmented, the model would learn "rotation
   fill artefact ⇒ hotdog" — a shortcut that looks like great validation accuracy and fails
   instantly in the real world. So *every* cached embedding is of an augmented image, negatives
   included.
2. **Moiré leakage.** The same trap one level up. If screen effects only ever appeared on hotdogs,
   the model would learn "moiré ⇒ hotdog". So screen effects are applied to every 7th hard
   negative too.

---

## Training in two stages

**Stage A — frozen base, cached embeddings.** MobileNetV2 runs over all 62,000 images *once* and
the 1280-d pooled vectors are cached to `.npz`. Head training then takes **~10 seconds on CPU**,
which makes hyperparameter sweeps essentially free. This is what turns a GPU-less machine from a
blocker into a non-issue.

**Stage B — fine-tune the top 30 layers** at LR 1e-5 on the live image pipeline. The only
genuinely slow step, and where the last points of accuracy live. Frozen ImageNet features get you
a usable model; they do not get you a good one.

### Threshold is a product decision

A false `HOTDOG` is both the funnier and the more damaging failure, so the operating point is
chosen by **false-positive rate** — the highest recall available while keeping FPR on the negative
classes at or under **2%** — rather than by maximizing F1.

Targeting FPR instead of precision matters on a skewed dataset. Precision moves with the
positive/negative ratio, so a precision target quietly encodes the dataset's shape into the
threshold; FPR is a property of the model alone. Switching from a 0.97-precision target to a 2%
FPR target lifted recall from 55% to 79% and F1 from 0.70 to 0.81 **without retraining anything** —
same weights, better-chosen operating point.

The chosen value lands in `models/metadata.json` and is read by both clients, so the number can't
drift between platforms.

---

## Results

Measured over the full 13,750-image held-out validation split.

| | Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Stage A (head only) | 0.8338 | 0.959 | 0.829 | 0.792 | 0.810 |
| Stage B (fine-tuned) | 0.4015 | **0.961** | **0.846** | 0.787 | **0.815** |
| Exported int8 `.tflite` | 0.4015 | 0.958 | 0.818 | 0.791 | 0.804 |

Quantization costs about 0.3 points of accuracy for a **70% smaller** file — 10.20 MB float32 down
to **3.07 MB** int8.

False positives by group, at the Stage B operating point:

| Group | Rate | Read |
| --- | --- | --- |
| Non-food (Imagenette) | **0 / 2,000** | Never fires on a dog, a chainsaw, or a church |
| Easy negative (other foods) | 23 / 5,000 | 0.5% |
| Hard negative (lookalikes) | 192 / 5,250 | 3.7% — where essentially all the error lives |

And recall: 782 / 1,000 plain hotdogs, plus **398 / 500 hotdogs photographed through a screen** —
the synthesized moiré training data works.

Worst lookalikes: `hamburger` 7.3%, `spring_rolls` 6.8%, `lobster_roll_sandwich` 6.7%,
`club_sandwich` 6.2%. Exactly the bread-adjacent, red/brown, roughly cylindrical foods you'd
predict, which is the reassuring outcome — the model is failing for legible reasons.

---

## Running it

```bash
cd training
uv sync
uv run python -m nothotdog.embed                      # Stage A cache (~10 min, one-time)
uv run python -m nothotdog.train --finetune-epochs 2  # Stage A head + Stage B
uv run python -m nothotdog.evaluate                   # confusion matrix + worst lookalikes
uv run python -m nothotdog.export                     # int8 .tflite + metadata.json
```

The first run downloads Food-101 (4.65 GiB) and Imagenette (~1.4 GiB) via TFDS.

```bash
cd web
npm install && npm run dev
```

`npm run sync:assets` (automatic on dev and build) copies the LiteRT WASM binaries out of
`node_modules` and the trained model out of `models/` into `public/`.

---

## Reading the evaluation

`training/artifacts/evaluation.json` breaks results down **by group**, not just overall. Overall
accuracy on a 10:1 dataset is a nearly useless number — a model that always says "not hotdog"
scores 89%. What matters is the per-group false-positive rate and the `worst lookalikes` table,
which names the specific foods the model mistakes for a hotdog. That table is the tuning signal:
if `lobster_roll_sandwich` dominates it, the answer is better augmentation, not more epochs.

---

## Four bugs worth documenting

Every one of these produced plausible-looking output while being wrong, which is the dangerous kind.

**1. The embedding cache is only valid while the base is frozen.** Scoring the head against cached
embeddings is correct for Stage A and *silently wrong* after fine-tuning — the base has moved, so
you're measuring a model that no longer exists. It reported 840 false positives where the real
model produces 215, making a genuine improvement look like a regression that nearly got reverted.
Evaluation now always runs the full model over real images.

**2. Keras exports a dynamic batch axis, and LiteRT.js rejects it.** `from_keras_model` yields
input shape `[-1,224,224,3]`; LiteRT.js requires an exact match and fails **at inference, not at
load**, so the model looks perfectly healthy until the first frame. Fixed by wrapping in
`Sequential([Input(batch_shape=(1, *IMAGE_SHAPE)), model])`. Converting a bare concrete function
instead fixes the shape but leaves unfrozen `READ_VARIABLE` nodes that fail to invoke.

**3. `Optimize.DEFAULT` outside the quantize branch makes the size comparison meaningless.** It
weight-quantizes the "float32" baseline too, so full int8 appeared **7% larger**. The real
comparison is 10.20 MB → 3.07 MB.

**4. Smoothing state must be scoped to continuous input.** Exponential smoothing is right for a
camera stream and wrong for a one-shot upload — and the state persisted across uploads, so a 60%
hotdog rendered `NOT HOTDOG` under a 40% threshold. Invisible to unit tests and invisible in
Python; only running real images through the real client exposed it.

The through-line: **test the artifact you ship, on the client you ship it on.** Three of these four
were invisible from inside the training code.

---

## Credits

Original app and approach by [Tim Anglade](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3).
MIT licensed.
