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

Measured over the full 13,750-image held-out validation split, after the dataset repair described
in bugs 5–7 below.

| | Threshold | Accuracy | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Stage B (fine-tuned, float32) | 0.4674 | **0.959** | 0.829 | **0.787** | **0.808** | — |
| Exported int8 `.tflite` | 0.5508 | 0.958 | **0.834** | 0.763 | 0.797 | **1.85%** |

The two thresholds differ because they are calibrated against two different models. Quantization
shifts the score distribution, so the operating point is re-picked on int8 scores by the same
FPR-capped policy rather than inherited from the float model — see bug 8.

Quantization costs about 0.15 points of accuracy for a **70% smaller** file — 10.20 MB float32
down to **3.07 MB** int8.

### Against the previously shipped model

The old artifact was trained on the corrupted hard-negative stream and shipped a float-calibrated
threshold, which put it at **2.15% FPR against its own advertised 2.0% cap**. Comparing headline
numbers against it directly is invalid — it bought recall with false positives it was not allowed
to spend. Forcing both models to a legal operating point with the same policy, on the same
corrected validation split:

| | old | new |
| --- | --- | --- |
| Accuracy | 0.9554 | **0.9576** |
| Precision | 0.8254 | **0.8344** |
| Recall | 0.7500 | **0.7627** |
| FPR | 1.94% | **1.85%** |

Better on every axis, at a lower false-positive rate.

False positives by group, at the Stage B operating point:

| Group | Rate | Read |
| --- | --- | --- |
| Non-food (Imagenette) | **0 / 2,000** | Never fires on a dog, a chainsaw, or a church |
| Easy negative (other foods) | 11 / 5,000 | 0.2% |
| Hard negative (lookalikes) | 233 / 5,250 | 4.4% — where essentially all the error lives |

And recall: 775 / 1,000 plain hotdogs, plus **406 / 500 hotdogs photographed through a screen** —
the synthesized moiré training data works.

Worst lookalikes: `hamburger` 9.2%, `breakfast_burrito` 7.6%, `lobster_roll_sandwich` 6.8%,
`french_fries` 6.4%. Exactly the bread-adjacent, red/brown, roughly cylindrical foods you'd
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

## Client parity

A client bug and a model change look identical from the outside: the number on screen is wrong.
`tests/parity/` pins down which one you are looking at. It holds six committed 224×224 PNGs
spanning the score range — two hotdogs, two hotdogs photographed on a screen, a lookalike, and an
easy negative — plus the score the shipped `.tflite` gives each one.

```bash
uv run --project training python scripts/parity.py
```

Run that, then drop the same files into the web app and compare. The reference is generated by
decoding the PNGs, exactly as a client does, rather than by scoring the float tensors the training
pipeline produces — the uint8 round-trip alone moves scores by a couple of points, and attributing
that to the client sends you hunting a bug that is not there.

Current agreement, WASM backend:

| Image | Python | Browser | Verdict |
| --- | ---: | ---: | --- |
| `hotdog-on-screen-b.png` | 86.72% | 86% | HOTDOG ✅ |
| `hotdog-a.png` | 44.92% | 42% | NOT HOTDOG ✅ |
| `hotdog-on-screen-a.png` | 42.19% | 42% | NOT HOTDOG ✅ |
| `hard-negative.png` | 34.77% | 37% | NOT HOTDOG ✅ |
| `hotdog-b.png` | 4.69% | 5% | NOT HOTDOG ✅ |
| `easy-negative.png` | 0.00% | 0% | NOT HOTDOG ✅ |

Every verdict agrees; scores drift by up to 0.03. The fixture doubles as a check that the
threshold in `metadata.json` actually reached the client — if `hotdog-on-screen-b.png` stops
reading HOTDOG, the asset sync broke.

Note that four of these six images are model failures: `hotdog-a`, `hotdog-b` and
`hotdog-on-screen-a` are hotdogs the model misses. That is deliberate. A fixture of easy wins
proves nothing, and these are the scores most likely to move when the model changes.

---

## Ten bugs worth documenting

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

**5. Branching a `tf.data` pipeline to partition it duplicates and drops elements.** The hard
negatives were split with `filter(i % 7 != 0)` and `filter(i % 7 == 0)` over the same dataset
object. tf.data re-executes the source once per branch, and TFDS with `shuffle_files=True`
reshuffles shard order on every iteration — so the two branches enumerated *different orders* and
the modulo partitioned two unrelated sequences. Measured on the validation hard negatives: **648
images appeared in both branches and another 648 appeared in neither**, collapsing 5,250 nominal
elements to 4,602 unique. The training set was quietly wrong for every run before this. Fixed by
making the source deterministic *and* deciding the split inside a single element pass with
`tf.cond`, so it stays correct even if someone re-enables shuffling.

**6. `shuffle_files=False` does not give you a deterministic dataset.** TFDS interleaves shards and
every `map` here uses `num_parallel_calls`; both reorder freely. Since augmentation seeds are
derived from the `enumerate` index, a reordered source silently reassigns every seed and every
screen decision. Needs an explicit `tf.data.Options()` with `deterministic = True`.

**7. A shuffle buffer cannot subsample a decoded image dataset.** `.shuffle(8192).take(n)` over
59k easy negatives looked fine but was structurally impossible: at 224×224×3 float32, a buffer
covering the source would need ~35 GB, and a small buffer biases toward whichever classes come
first. It only ever worked by accident, because the reshuffling source in bug #5 pre-mixed it.
Replaced with Bresenham index striding — deterministic, exact, and free.

**8. A threshold calibrated on float scores does not survive quantization.** The operating point
was picked from the Keras model and copied verbatim into the int8 model's metadata. Quantization
moves the score distribution, so the advertised 2.0% false-positive cap was really **2.15%**. The
export step already ran the full split through the int8 interpreter, so it had every score needed
to recalibrate — it just wasn't using them.

**9. `sin(n·π)` is zero for every integer n.** The screen effect's scanline term was
`tf.sin(ys * math.pi)` where `ys` is an integer pixel row. It had been contributing exactly nothing
since it was written. Dead code that reads as live code, in a file with no assertions on its
output.

**10. A regression gate that compares metrics across operating points reports fiction.** Precision
and recall are a single point read off an ROC curve, and the gate diffed them against whatever the
last build happened to record. After bug 8 was fixed, the retrained model — better on accuracy,
precision, recall *and* false-positive rate when both are held at a legal operating point — was
flagged as a 0.028 recall regression, purely because the baseline had been measured at an illegal
2.15% FPR. The gate already refused to diff builds that scored different images; it now applies the
same reasoning to builds read at different points on the curve. It stays armed for everything
after: every build records its rate, and the absolute cap is checked independently, so a model
cannot dodge the comparison by quietly moving its threshold.

Bug #5 has a mobile twin worth its own line: `vision-camera-resize-plugin` returns `float32` in
`[0,1]`, but the normalization constants in `metadata.json` are stated in the byte domain `[0,255]`.
Applying them directly pinned every pixel to −1.0, so the model saw a black frame on **every single
camera frame**. That path had never worked. The fix keeps reading `scale`/`offset` from metadata and
maps into the byte domain first, rather than hardcoding `p * 2 - 1`.

The through-line: **test the artifact you ship, on the client you ship it on.** Most of these were
invisible from inside the training code, and the worst three were invisible from anywhere — they
needed someone to count elements and compare orders rather than read the code and nod.

---

## Known limitations

**The validation split does triple duty.** It drives early stopping, threshold selection, and the
final reported metrics. That makes the headline numbers optimistic: the threshold is chosen on the
same images it is then scored against. A proper three-way split is the right fix. It is not done
here, so read the reported precision and recall as an upper bound rather than an estimate of
field performance.

**Web inference is forced to WASM.** The WebGPU backend disagreed with the Python reference by up
to 0.082 on identical input tensors, which is enough to flip a verdict near the threshold. WASM is
far closer but not exact: it tracks Python to within 0.03 (see [Client parity](#client-parity)).
Correctness over speed, but it is a real performance cost that has not been benchmarked in the
camera loop.

**Browser and Python scores differ by up to 0.03.** Preprocessing is not the cause — the canvas
pipeline is byte-identical to Python's decode, verified by comparing pixel checksums of the same
files in both. The residual is the inference engine: TFLite's XNNPACK CPU kernels and LiteRT.js's
WASM kernels round int8 requantization differently, and that difference compounds through
MobileNetV2's depth. It has never flipped a verdict on the fixture, where the nearest image sits
0.10 from the threshold, but an image landing within ~0.03 of 0.55 could read differently on the
two runtimes.

**Two open Dependabot alerts with no fix available.** `image-size` has DoS advisories against its
ICNS, JXL and HEIF parsers with a vulnerable range of `<= 2.0.2` — which is the latest published
version, so `first_patched_version` is `null` and upgrading is a no-op. It reaches the tree as a
transitive dependency of Metro, which is a `devDependency`: it runs at bundle time and never ships
in the app binary. Triggering it would mean adding a hostile image to your own assets. Left open
rather than dismissed, so it resurfaces when a patch lands.

---

## Credits

Original app and approach by [Tim Anglade](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3).
MIT licensed.
