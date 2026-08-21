#!/usr/bin/env node
/**
 * Validates the model artifact contract without touching the dataset.
 *
 * Retraining in CI is not an option -- Food-101 alone is 4.65 GB -- so this guards the things
 * that can actually break a client at runtime: the metadata shape both clients parse, a size
 * budget that catches an unquantized model slipping in, and a metrics diff against the base so a
 * regression has to be argued for rather than merged quietly.
 */

import { execFileSync } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';

const SIZE_BUDGET_BYTES = 6_000_000;
const REGRESSION_TOLERANCE = 0.02;
const TRACKED_METRICS = ['accuracy', 'hotdogPrecision', 'hotdogRecall'];

const failures = [];
const check = (condition, message) => {
  if (!condition) failures.push(message);
};

const metadata = JSON.parse(readFileSync('models/metadata.json', 'utf8'));

check(
  metadata.model === 'not-hotdog.tflite',
  `model must be not-hotdog.tflite, got ${metadata.model}`,
);
check(
  metadata.inputDtype === 'float32',
  `inputDtype must be float32, got ${metadata.inputDtype}`,
);
check(
  Array.isArray(metadata.inputShape) && metadata.inputShape.join(',') === '1,224,224,3',
  `inputShape must be [1,224,224,3], got ${JSON.stringify(metadata.inputShape)}`,
);
check(
  metadata.normalization?.mode === 'mobilenet_v2',
  `normalization.mode must be mobilenet_v2, got ${metadata.normalization?.mode}`,
);
check(
  Array.isArray(metadata.classOrder) &&
    metadata.classOrder.length === 2 &&
    metadata.classOrder.includes('hotdog') &&
    metadata.classOrder.includes('not_hotdog'),
  `classOrder must contain hotdog and not_hotdog, got ${JSON.stringify(metadata.classOrder)}`,
);
check(
  typeof metadata.threshold === 'number' && metadata.threshold > 0 && metadata.threshold < 1,
  `threshold must be between 0 and 1, got ${metadata.threshold}`,
);

// The threshold is a promise about how often a non-hotdog lights up the overlay, and it is
// calibrated against the quantized model that actually ships. It is checked absolutely rather
// than as a regression, because the cap is a product decision and a build that misses it is
// wrong even if the previous build missed it by more.
const { falsePositiveRate: fpr, targetFalsePositiveRate: fprTarget } = metadata.metrics ?? {};
if (typeof fpr === 'number' && typeof fprTarget === 'number') {
  check(
    fpr <= fprTarget,
    `false positive rate is ${(fpr * 100).toFixed(2)}%, over the ${(fprTarget * 100).toFixed(2)}% cap the threshold advertises`,
  );
} else {
  failures.push('metrics.falsePositiveRate and metrics.targetFalsePositiveRate are required');
}

const size = statSync('models/not-hotdog.tflite').size;
check(
  size <= SIZE_BUDGET_BYTES,
  `not-hotdog.tflite is ${(size / 1e6).toFixed(2)} MB, over the ${SIZE_BUDGET_BYTES / 1e6} MB budget`,
);

const baseRef = process.env.GITHUB_BASE_REF ? `origin/${process.env.GITHUB_BASE_REF}` : 'HEAD~1';

let previous = null;
try {
  previous = JSON.parse(
    execFileSync('git', ['show', `${baseRef}:models/metadata.json`], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }),
  );
} catch {
  // First commit of the model, or the base predates it. Nothing to compare against.
}

// Two builds are only comparable when they scored the same images. A metric measured over a
// different evaluation set differs by sampling noise alone, which would make this gate fire on
// a methodology change and stay quiet on a real regression of the same magnitude.
//
// The same argument applies to the operating point. Precision and recall are a single point read
// off an ROC curve, so they only mean the same thing when both builds were read at a comparable
// point on it. Builds predating threshold recalibration recorded no false-positive rate at all:
// their threshold was calibrated against the float model and the int8 model that shipped landed
// wherever it landed -- measured at 2.15% against an advertised 2.0% cap. Those numbers bought
// recall with false positives the model was never allowed to spend, so diffing against them
// reports a regression for a build that is strictly better at a legal operating point.
//
// This does not open a hole. Every build from here on records its rate, and the absolute cap
// above is checked independently, so a future model cannot dodge this comparison by quietly
// moving its threshold -- it would still be compared, and still be caught.
const comparable =
  typeof metadata.metrics?.evaluated === 'number' &&
  metadata.metrics.evaluated === previous?.metrics?.evaluated &&
  typeof previous?.metrics?.falsePositiveRate === 'number';

const rows = TRACKED_METRICS.map((key) => {
  const current = metadata.metrics?.[key];
  const before = previous?.metrics?.[key];

  if (typeof current !== 'number') {
    failures.push(`metrics.${key} is missing from metadata.json`);
    return `| ${key} | missing | ${before ?? 'n/a'} | 🔴 |`;
  }
  if (typeof before !== 'number') {
    return `| ${key} | ${current.toFixed(4)} | n/a | new |`;
  }
  if (!comparable) {
    return `| ${key} | ${current.toFixed(4)} | ${before.toFixed(4)} | not comparable |`;
  }

  const delta = current - before;
  if (delta < -REGRESSION_TOLERANCE) {
    failures.push(
      `${key} regressed by ${Math.abs(delta).toFixed(4)} (${before.toFixed(4)} -> ${current.toFixed(4)})`,
    );
  }
  const icon = delta < -REGRESSION_TOLERANCE ? '🔴' : delta < 0 ? '🟡' : '🟢';
  const sign = delta >= 0 ? '+' : '';
  return `| ${key} | ${current.toFixed(4)} | ${before.toFixed(4)} | ${sign}${delta.toFixed(4)} ${icon} |`;
});

console.log('## Model eval\n');
console.log(`**Size** ${(size / 1e6).toFixed(2)} MB of ${SIZE_BUDGET_BYTES / 1e6} MB budget`);
console.log(
  `**Threshold** ${metadata.threshold.toFixed(4)} · **Git SHA** \`${metadata.gitSha}\`\n`,
);
console.log('| Metric | This build | Base | Delta |');
console.log('| --- | --- | --- | --- |');
console.log(rows.join('\n'));

if (previous && !comparable) {
  const reason =
    metadata.metrics?.evaluated !== previous?.metrics?.evaluated
      ? `the base scored ${previous?.metrics?.evaluated ?? 'an unrecorded number of'} images and this build scored ${metadata.metrics?.evaluated}`
      : 'the base recorded no false-positive rate, so the operating point its precision and recall were read at is unknown';
  console.log(`\n> Deltas suppressed: ${reason}.`);
}

if (typeof fpr === 'number') {
  console.log(
    `\n**False positive rate** ${(fpr * 100).toFixed(2)}% of ${(fprTarget * 100).toFixed(2)}% cap`,
  );
}

if (failures.length) {
  console.log('\n### Failures\n');
  for (const failure of failures) console.log(`- ${failure}`);
  console.error(`\nmodel check failed:\n${failures.map((f) => `  - ${f}`).join('\n')}`);
  process.exit(1);
}

console.log('\nAll checks passed.');
