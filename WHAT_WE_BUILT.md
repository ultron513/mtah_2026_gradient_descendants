# What We Built — Gradient Descendants

**Competition:** More than a Hack 2027, Phase 1 — ball possession (FC Barcelona vs Athletic Club)
**Current system:** Phase 9
**Team:** Gradient Descendants

## Task

For every video frame, predict who controls the ball: H (Barcelona), A (Athletic Club), D (dead ball), or X (ungraded/abstain). The ranking metric is spellF1, with macroF1 as a secondary metric.

## Data, assets, and rules

- **Competition match:** BAR-ATH tracking data, Smart Tagging, and the public five-minute labels are used to build and check the pipeline. The public window is a sanity check only; it is not used for model training or tuning. Human frame checks were used only to audit decisions, never to create labels.
- **Training data:** seven labelled 2022–23 Bundesliga / 2. Bundesliga matches from IDSSE/DFL. The dataset provides per-frame BallStatus and BallPossession labels from DFL positional data. Source: Bassek, Weber, Rein & Memmert (2025), [An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y). No labels from the provided BAR-ATH match are used for training.
- **Possession harmonisation:** external possession can switch on looser contact than the competition's controlled-possession rule. Within an alive stretch, a switch is accepted only when the other team's run lasts at least T seconds; shorter runs remain with the previous team. The first run after a restart is kept and dead-ball labels are unchanged. Models train with T = 1 s and are evaluated at T = 0, 0.5, 1, and 2 s.
- **Output contract:** one CSV row per frame, numbered 0–199318. Graded frames use H/A/D; ungraded frames use X.

## Pipeline

1. Parse the tracking and tagging files into a shared match representation and reproduce the competition scorer.
2. Measure tracking degradation from BAR-ATH, then degrade clean IDSSE matches so their tracking resembles the competition feed.
3. Compute ball geometry, player/team shape and movement, ball visibility, and dead-ball evidence.
4. Fit NumPy logistic-regression and custom histogram gradient-boosting evidence models.
5. Decode frame scores into whole possession sequences with Potts and semi-Markov methods. Stoppage-edge signals and kick-off tags refine boundaries.
6. Select settings by leave-one-match-out evaluation on simulated IDSSE matches, with the four possession-control thresholds represented in the objective.

The pipeline also includes a scorer reimplementation, a format validator, plausibility checks, and deterministic fixed-point decoding. Match data and generated CSVs stay outside the public repository.

## Phase 9: realistic ball simulation

The earlier simulator dropped the ball at similar overall rates throughout play. BAR-ATH instead sees the ball on about 87% of live-play frames and about 18% of stoppage frames. Phase 9 profiles this pattern from the Phase 6 model's predictions, not hand labels, and measures stoppage visibility, live-play gaps, and gaps that extend across stoppages and restarts.

The simulator now smooths the ball path, carries the ball toward the restart point and lets it rest with small handling movements, applies measured visibility patterns during stoppages, and adds live-play gaps to target 88% visibility. Player simulation, model features, and decoder design stayed the same; the models were retrained and settings reselected on the new simulations. The Phase 6 world remains available as a reproducible fallback.

## Results recorded by the project

| Evaluation | Phase 6 | Phase 9 |
|---|---:|---:|
| Mean spellF1 across seven held-out matches | 0.5570 | 0.5969 (+0.040; Phase 9 better on 7/7) |
| Blind real-frame audit | 25/53 correct | 32/53 correct |
| Public five-minute check: spellF1 / macroF1 | 0.600 / 0.727 | 0.562 / 0.826 |

The latest README also reports a pooled leave-one-match-out objective of 0.5972; the Phase 9 spec's 0.5969 is the arithmetic mean of the seven per-match values. The public five-minute window was not used for tuning and contains only 17 possessions, so its spellF1 change is noisy.

The Phase 9 submission's recorded SHA-256 prefix is `28f7229f96045c50`; the Phase 6 fallback prefix is `edde29db13a56e2d`. The validator prints the full SHA-256.

## Experiments not adopted

Phase 8 tried receiver/co-movement features for short possessions, training on both smooth and sharp simulations, and averaging several models. None improved at least six of seven held-out matches, so they are not part of the selected pipeline. Camera-based ball detection was also evaluated and not adopted because tracking already sees the ball in most live play and the detections were noisier.

## Limitations

- A ball held or placed during a throw-in or free kick can still be called live.
- A long missing-ball gap that spans a stoppage and the restart can be called dead for too long.
- Live-play visibility is set to 88%; the Phase 9 notes recommend checking sensitivity across the estimated 87–94% range.
- The public-window spellF1 is lower than Phase 6, despite the higher macroF1; the sample is too small to tune against.

## Reproduce the submission

Use the official data on your machine and keep `MTAH_STATE_OVERRIDES` unset for the recorded Phase 9 settings and hash.

```powershell
# Stage 1: regenerate the Phase 6 labels used to profile BAR-ATH state
python -m possession noise-profile
python -m possession simulate --world phase6
python -m possession phase1-select
python -m possession predict --out stage1
Copy-Item 'stage1/Gradient Descendants.csv' 'cache/stage1_labels.csv'

# Stage 2: train and predict with the Phase 9 simulator
python -m possession state-profile --labels cache/stage1_labels.csv
python -m possession simulate
python -m possession phase1-select
python -m possession predict
python validate_submission.py 'submissions/Gradient Descendants.csv'
```

See [README.md](README.md) for setup and the complete run instructions, and [docs/specs/phase9.md](docs/specs/phase9.md) for the full Phase 9 evidence. The changelog records the full suite as 64 passed and 1 skipped.

## Submission items

1. **Possession CSV:** `Gradient Descendants.csv`, generated and validated locally, then uploaded separately.
2. **Technical document:** this Markdown overview or a short PDF covering the approach, assets and training data, possession rules, limitations, and reproduction.
3. **Public repository:** [ultron513/mtah_2026_gradient_descendants](https://github.com/ultron513/mtah_2026_gradient_descendants).