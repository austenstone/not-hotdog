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

A false `HOTDOG` is both the funnier and the more damaging failure, so the operating point targets
**0.97 precision** on the hotdog class and maximizes recall subject to that — rather than
maximizing F1. The chosen value is written into `models/metadata.json` and read by both clients,
so the number can't drift between platforms.

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

## Credits

Original app and approach by [Tim Anglade](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3).
MIT licensed.
