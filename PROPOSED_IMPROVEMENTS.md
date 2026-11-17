# Phased Implementation Roadmap (Step-by-Step)

**Team Gradient Descendants** · *More than a Hack 2027, Phase 1 (Ball Possession: BAR–ATH)*  
**Starting baseline (when drafted):** Phase 6 (`spellF1` 0.592 on simulated IDSSE, 0.600 on the public five-minute slice)  
**Original objective:** Evaluate whether disciplined, non-breaking changes could improve short-possession performance and raise `spellF1` toward the theoretical target (~0.84). This was an aspiration, not a measured outcome or guarantee.

> **Status note — 17 November 2026:** This is a historical roadmap drafted against the Phase 6 baseline. The repository is now at Phase 9. Steps 1–7 were evaluated; the Phase 7 changelog records that their experimental behaviors were kept behind off-by-default controls after they failed the adoption rule. Phase 8 experiments were not adopted. The Phase 9 game-state-aware simulator was adopted. This file preserves the earlier hypotheses and evaluation process; its imperative wording is not a current implementation checklist. For the current implementation and results, see [CHANGES.md](CHANGES.md) and the [Phase 9 specification](docs/specs/phase9.md).

---

## Guiding Principles & Safety Protocol

1. **Step-by-Step Isolation:** Each step is implemented and validated independently before the next step begins.
2. **Strict Backwards Compatibility:** Step 1 and subsequent additions must preserve existing function signatures, default arguments, and file formats so that existing baseline tests (`tests/test_phase0.py` through `tests/test_phase6.py`) remain 100% green.
3. **Pure NumPy & Deterministic:** No Pandas or heavy runtime dependencies; bit-exact integer/fixed-point arithmetic for decoding.
4. **Validation Guardrails:** Every change must satisfy:
   - Leave-one-match-out cross-validation across all 7 IDSSE games.
   - Non-regression across all 4 control definitions ($0\text{s}, 0.5\text{s}, 1\text{s}, 2\text{s}$).
   - Strict paired win check ($\ge 6\text{ of } 7$ matches).
5. **Mandatory Live Logging in `CHANGES.md`:** 
   Whenever the agent implements any step or code update described in this roadmap, the agent **must immediately update [`CHANGES.md`](CHANGES.md)** with:
   - Target file modified
   - Exact lines added or changed (`L<start>-L<end>`)
   - The specific role/purpose of the change
   - Backwards compatibility check
   - Verification commands and test results (e.g. `pytest` output)

---

## Step 1: Simulator Smoothing Calibration & Conservative Stoppage Ball

> **Focus:** Zero impact on production models; eliminates the gap between simulated data and real-world Metrica tracking without breaking any existing pipeline.

### Objectives
1. **Calibrate Ball Smoother:** The real Metrica tracker smooths the ball internally ($\sigma \approx 7$ frames). Apply this matching Gaussian filter in `possession/noise/simulate.py` so training games match BAR–ATH kinematics.
2. **Conservative Stoppage Ball:** Replace abrupt ball freeze/teleportation during dead-ball intervals with a bounded spatial walk tied to walking velocities, avoiding the full-match false stoppages encountered in earlier trials.

### Implementation Details
* **Target File:** [`possession/noise/simulate.py`](possession/noise/simulate.py)
* **Function:** `_degrade_ball()` and `carry_dead_ball()`
* **Non-Breaking Guarantee:** Exposed via an optional keyword argument `smooth_sigma=0.0` (default) which maintains identical behavior for legacy tests, while setting `smooth_sigma=7.0` for calibrated simulations.
* **Pass Criterion:** Simulated ball velocity distribution matches BAR–ATH's KS-test threshold, and existing synthetic test suite passes without errors.

---

## Step 2: Asymmetric Decoder Restart Penalties

> **Focus:** Low-risk decoding enhancement; removes the penalty barrier for short possessions immediately following dead-ball restarts.

### Objectives
1. **Differentiate Transition Costs:** In `possession/decode.py`, keep the turnover penalty ($H \leftrightarrow A$) high to prevent in-play flicker, but lower the entry penalty for transitions out of Dead ball ($D \to H$ and $D \to A$) at detected restart boundaries.
2. **Capture Immediate Post-Restart Spells:** Allow genuine brief possessions right after a set piece/restart to be credited rather than suppressed by the symmetric transition tax.

### Implementation Details
* **Target File:** [`possession/decode.py`](possession/decode.py)
* **Function:** `potts_edges()`
* **Non-Breaking Guarantee:** Keep default `lam_dead_out=None` (falling back to standard symmetric `lam_dead`), ensuring exact reproducibility of Phase 6 results unless explicitly enabled.
* **Pass Criterion:** Increased possession count toward ground truth (~5,100) on simulated validation games without degrading macroF1.

---

## Step 3: Targeted Micro-Context Windows for Kinematic Signals

> **Focus:** Selective high-frequency feature extraction that avoids the noise of blanket short-window averaging.

### Objectives
1. **Targeted Micro-Windows ($0.4\text{s}$ / 10 frames):** Earlier experiments showed blanket $0.5\text{s}$ averaging across *all* position features failed the $\ge 6/7$ match safety check due to tracking flicker. We apply the 10-frame ($0.4\text{s}$) window **only** to derivative signals: ball acceleration and heading change.
2. **Preserve Macro Stability:** Maintain the existing $\pm 2\text{s}$ (50 frames) and $\pm 5\text{s}$ (125 frames) windows that anchor continuous possessions.

### Code Implementation Instruction: Commented Toggle Style
When adding the $0.4\text{s}$ micro-window code into `possession/features.py`, the agent/developer **must implement it in a commented-out toggle format** so the codebase can be tested in two distinct modes without code rewrites:

* **Pass 1 (Baseline Run — 2s & 5s Only):**
  The codebase runs strictly with the existing, proven $\pm 2\text{s}$ and $\pm 5\text{s}$ windows. This verifies zero regressions and zero additional noise.
* **Pass 2 (Experimental Run — 2s, 5s + 0.4s Micro-Window):**
  The $0.4\text{s}$ code is kept inside a clean, commented block. To test the extra sensitivity and check whether the added high-frequency noise is acceptable under the $\ge 6/7$ match safety rule, the user or agent simply removes the comment markers.

#### Required Code Structure in `possession/features.py`:
```python
# =========================================================================
# STEP 3: CONTEXT WINDOW CONFIGURATION
# =========================================================================

# --- MODE 1: Standard Macro Windows (2s and 5s only) [ACTIVE BY DEFAULT] ---
# Keeps the proven Phase 5 baseline completely intact with zero added noise.
# CONTEXT_SECONDS remains (2, 5).

# --- MODE 2: Experimental 0.4s (10-frame) Micro-Window [COMMENTED OUT] ---
# Uncomment the block below to run evaluation with 0.4s micro-windows enabled
# to assess whether the added turnover sensitivity beats the noise threshold.
#
# def compute_micro_context_04s(f, fps):
#     """Compute 10-frame (0.4s) rolling context for ball heading and acceleration."""
#     micro = {}
#     w = int(round(0.4 * fps))  # 10 frames at 25 fps
#     known = f["ball_known"] > 0
#     for name in ("ball_speed", "heading_diff", "vel_diff"):
#         if name in f:
#             x = np.where(known, f[name], np.nan)
#             micro[f"ctx_{name}_p04"] = np.nan_to_num(_window_mean(x, -w, 0))
#             micro[f"ctx_{name}_f04"] = np.nan_to_num(_window_mean(x, 1, w + 1))
#     return micro
#
# # In compute(match):
# # base.update(compute_micro_context_04s(base, fps))
# =========================================================================
```

### Implementation Details
* **Target File:** [`possession/features.py`](possession/features.py)
* **Function:** `context_windows()` and `compute()`
* **Non-Breaking Guarantee:** By keeping the $0.4\text{s}$ block commented out initially, existing tests and baseline results run 100% unaltered.
* **Pass Criterion:** Wins on $\ge 6 \text{ of } 7$ validation games under Leave-One-Match-Out evaluation when uncommented.

---

## Step 4: Kinematic Impulse Features & Relative Closing Velocity

> **Focus:** Instantaneous micro-touch identification to resolve turnovers $< 2\text{s}$.

### Objectives
1. **Ball Heading Discontinuity ($\pm 6$ Frames Baseline):** Flag sharp angle kinks ($> 60^\circ$) measured across a $\pm 6$ frames baseline ($\pm 0.24\text{s}$, total span of 12 frames $\approx 0.48\text{s}$).
   * **Vector Formulation:** Measure direction change between inbound and outbound vectors:
     $$\vec{v}_{\text{in}} = \vec{x}_t - \vec{x}_{t-6}, \qquad \vec{v}_{\text{out}} = \vec{x}_{t+6} - \vec{x}_t$$
     $$\cos(\Delta \theta_t) = \frac{\vec{v}_{\text{in}} \cdot \vec{v}_{\text{out}}}{\|\vec{v}_{\text{in}}\| \|\vec{v}_{\text{out}}\|}$$
   * **Why $\pm 6$ frames:** Matches the natural $\sim 0.5\text{s}$ micro-window and straddles the optical tracker's internal smoothing kernel ($\sigma \approx 7$ frames) to prevent derivative blunting, while staying tight enough to ignore aerodynamic curving of crosses/passes.
   * **Proximity condition:** Active only when opponent distance $d_{\text{opp}} \le 2.0\text{ m}$.
2. **Player Deceleration & Turn Concurrency:** Pair ball direction shifts with receiver acceleration dips ($a_{\text{player}} < -2.0 \text{ m/s}^2$) and turns to isolate intentional touches from balls passing nearby.
3. **Relative Closing Velocity ($v_{\text{rel}}$):** Replace static scalar distance with signed radial velocity to indicate active interception.

### Implementation Details
* **Target File:** [`possession/features.py`](possession/features.py)
* **New Constant:** `KINK_SPAN = 6` (leaves existing `SPEED_SPAN = 5` completely untouched).
* **Functions:** New helpers `_ball_angle_kink()`, `_player_impulses()`, and `_closing_velocity()`
* **Non-Breaking & Zero Ripple Effect Guarantee:**
  * Self-contained helper functions; pure NumPy vector operations without memory allocations inside loops.
  * Adds an isolated key `f["ball_kink_6f"]` to the feature dictionary; existing feature keys and Phase 0–6 design matrices remain completely unchanged.
  * Robust boundary handling: pads with `0.0` for boundary frames ($t < 6$ or $t \ge T - 6$) and across missing ball gaps so no NaNs leak into GBDT trees.
* **Pass Criterion:** Improves team classification accuracy on in-play frames $< 2\text{s}$ from ~50% to $> 65\%$.

---

## Step 5: Soft Pitch Boundary Features & Weak Tactical Set-Piece Priors

> **Focus:** Replace fragile hard rules with soft features and probabilistic priors, preventing false stoppage triggers from 3D camera parallax and automated tagging timing jitter.

### Objectives
1. **Continuous Soft Pitch Margin Feature (Avoiding Parallax False Stoppages):**
   * **The 3D Parallax Danger:** 2D planar homography assumes $z = 0$ (ground plane). An airborne pass or cross travelling legally inside the pitch at height $z > 0$ projects *outside* the sideline due to camera angle perspective. A hard out-of-bounds rule ($x \notin [0,1] \implies D$) would fatally kill live aerial plays.
   * **Soft Feature Formulation:** Instead of a hard clamp, compute continuous distance beyond the line:
     $$\text{boundary\_margin} = \max(0, -x, x-1, -y, y-1)$$
   * Feed this as a continuous feature into the dead-ball GBDT model, allowing the tree to evaluate boundary distance in context with player speeds and ball acceleration rather than hard-killing frames.
2. **Weak Soft Prior for Automated Smart Tagging (~50–60% Coverage):**
   * **The Automated Tagging Limitation:** Smart Tagging is an automated heuristic covering only ~50–60% of real match restarts, with start/end timing jitter of $\pm 1\text{s}$ to $3\text{s}$ relative to actual whistle/restart frames.
   * **Weak Prior Formulation:** Do not force hard state constraints. Instead, add a gentle log-odds bias ($+0.5$ to $+0.8$ nats) to the dead-ball emission and Phase 6 restart edge detector during tagged windows.
   * **Safety Mechanism:** Strong kinematic tracking evidence (e.g. players sprinting, passing) can easily override the weak prior, while ambiguous frames safely receive the slight nudge toward dead ball.

### Implementation Details
* **Target Files:** [`possession/features.py`](possession/features.py) & [`possession/emission.py`](possession/emission.py)
* **Functions:** `boundary_margin` calculation in `compute()` and optional `tagging_prior` addition in `emission.py`.
* **Non-Breaking Guarantee:** If homography or tagging XML files are omitted, `boundary_margin` defaults to `0.0` and `tagging_prior` defaults to `0.0`, preserving 100% compatibility with standalone synthetic benchmarks.
* **Pass Criterion:** Improves macroF1 on stoppage transitions without fracturing aerial cross possessions or missing untagged restarts.

---

## Step 6: Signed Touch State & Possession Persistence Features

> **Origin (Teammate Review):**  
> *"Acha jo 0/1 flag ta karichu seta +1/-1 kari haba ki touching player team pain, and maybe then team of last touch and time since last touch add karile ig pura possession cover kari habi rather than 1 frame."*  
> **Translation (Odia/English mix):** *"The 0/1 touch flag we created — make it +1/−1 based on the touching player's team. Then add 'team of last touch' (forward-filled) and 'time since last touch'. That way we'd cover the entire possession span rather than just a single frame."*

> **Focus:** Extend the single-frame ball kink detector (Step 4) into a **stateful possession signal** that persists across pass flights, ball gaps, and contested moments — covering the full possession span rather than firing on only 1 frame.

### The Problem Being Solved
Step 4's kink detector fires at the **instant** a player touches the ball ($1$ frame). But a pass takes $1$–$3$ seconds ($25$–$75$ frames) to travel across the pitch. During that flight:
- Neither team is within $2\text{m}$ of the ball.
- The instantaneous kink flag drops back to `0`.
- The tree model has no direct evidence of who struck the pass.

### Feature Design (3 Components)

1. **Signed Touch Event** ($T_t \in \{+1, -1, 0\}$):
   Instead of a binary $0/1$ kink flag, sign it by the touching player's team:
   $$T_t = \begin{cases} +1 & \text{Home player nearest at kink frame} \\ -1 & \text{Away player nearest at kink frame} \\ 0 & \text{no kink detected or ambiguous} \end{cases}$$
   Follows the codebase convention where all team differentials use $+$ = Home, $-$ = Away (matching `team_margin`, `vel_diff`, `push_diff`, `ctrl_run`).

2. **Team of Last Touch** ($L_t$, forward-filled state):
   Forward-fill the most recent non-zero $T_t$ value across subsequent frames:
   $$L_t = T_{\tau} \quad \text{where} \quad \tau = \max\{s \le t : T_s \ne 0\}$$
   This carries possession ownership through the entire pass flight until the next detected touch event.

3. **Time Since Last Touch** ($\Delta t$, seconds):
   $$\Delta t = (t - \tau) / \text{fps}$$
   Fed directly into the GBDT as an explicit input so the tree can learn to discount stale touches. Paired with an exponential decay weighting:
   $$\text{touch\_signal}(t) = L_t \times \exp(-\Delta t / 1.5)$$
   After $\sim 1.5\text{s}$ the signal drops by $> 63\%$, forcing the model to rely on team geometry for longer gaps.

### Empirical Validation (Old Simulation)
Tested on old simulation data, this design improved short-possession team accuracy:
> **$< 2\text{s}$ possession accuracy: 0.50 → 0.59** (+9 percentage points)

This confirms the risks are under control — if errors were dominating, accuracy would have dropped.

### Risk Analysis & Conditioning Guardrails

Three specific risks were identified and are mitigated by mandatory conditioning logic:

#### Guard 1: Deflection / Block Filter (Risk #1, Score 21/100)
* **Risk:** Defenders constantly block passes, deflect crosses, or poke the ball during tackles without gaining control. The kink detector fires, and forward-filling would hallucinate micro-possessions.
* **Why Impact Is Low:** The IDSSE training labels already encode the competition's "controlled possession" rule. Trees trained on thousands of deflections labelled as "no possession change" learn to ignore kinks not followed by sustained proximity.
* **Mandatory Conditioning:**
  ```python
  # Only assign a signed touch when the nearest player has sustained proximity.
  # A transient deflection (opponent within 2m for < 3 frames) is zeroed out.
  MIN_PROXIMITY_FRAMES = 3  # 0.12s at 25 fps
  ```

#### Guard 2: False Touch Carry Limiter (Risk #2, Score 20/100)
* **Risk:** If tracking jitter creates a false kink, raw forward-fill propagates the error for 75+ frames (3 seconds).
* **Why Impact Is Low:** The exponential decay ($\exp(-\Delta t / 1.5)$) and explicit `time_since_last_touch` as a tree input cap the blast radius to ~1.5s. The tree learns that old touches are unreliable.
* **Mandatory Conditioning:**
  ```python
  # Exponential decay prevents stale errors from persisting.
  touch_signal = last_touch_team * np.exp(-time_since_touch / 1.5)
  # Hard cap: zero out touch memory after MAX_CARRY seconds.
  MAX_CARRY_S = 5.0  # Beyond 5s the touch is ancient history.
  touch_signal[time_since_touch > MAX_CARRY_S] = 0.0
  ```

#### Guard 3: Contested Scramble Margin Gate (Risk #3, Score 18/100)
* **Risk:** In 50/50 duels, $\pm 0.2\text{m}$ tracker noise can misattribute the nearest player's team.
* **Why Impact Is Low:** Weighting touch confidence by the distance margin sends equal-distance cases to $\approx 0$.
* **Mandatory Conditioning:**
  ```python
  # Only assign non-zero touch team if clear separation exists.
  MIN_SEPARATION_M = 0.5  # metres
  margin_at_touch = d_opponent - d_team
  # If both teams are equidistant (< 0.5m gap), set touch to 0 (ambiguous).
  touch_team[margin_at_touch < MIN_SEPARATION_M] = 0
  ```

### Additional Safety: Dead-Ball Reset
`last_touch_team` and `time_since_last_touch` **must reset to 0** at:
- All detected stoppage edges (Phase 6 edge detector).
- Period boundaries (halftime, extra time).
- Frames where `ball_present == False` for $> 2\text{s}$ continuously.

### Implementation Details
* **Target File:** [`possession/features.py`](possession/features.py)
* **New Constants:** `MIN_PROXIMITY_FRAMES = 3`, `MIN_SEPARATION_M = 0.5`, `MAX_CARRY_S = 5.0`, `TOUCH_DECAY_TAU = 1.5`
* **New Helpers:** `_signed_touch()`, `_last_touch_state()`, `_time_since_touch()`
* **New Feature Keys:** `f["touch_team"]`, `f["last_touch_team"]`, `f["time_since_touch"]`, `f["touch_signal"]`
* **Non-Breaking Guarantee:** Purely additive feature columns; existing feature dictionary keys and Phase 0–6 design matrices are completely untouched.
* **Pass Criterion:** Short-possession ($< 2\text{s}$) team accuracy improves from ~50% toward ~59%+ on simulated IDSSE games while passing the $\ge 6/7$ match paired check.

---

## Step 7: Touch Detector Recall & Team Ownership Correction

> **Origin:** Teammate review of the Step 6 touch-state implementation.

### Problem

The Step 6 touch features pass the available tests but fire too rarely to affect the model. BAR–ATH produces about 0.05 detected touches per minute and has a non-zero touch state on only about 0.3% of in-play frames. The detector currently requires all of these conditions simultaneously: a ball kink above 60 degrees, player deceleration below −2 m/s², a turn above 30 degrees, proximity within 2 m for three frames, the same nearest player throughout, and at least 0.5 m separation from the other team. The smoothed tracking ball often removes the kink and movement evidence.

### Ownership issue

The candidate-touch gate currently requires the player to belong to the team opposite the previous nearest team. This prevents ordinary same-team touches from becoming signed events. The later signed-touch stage cannot recover those missing home or away touches.

### Resolution

1. Remove the opponent-only eligibility gate so either team can produce a signed touch.
2. Keep the safety guards: three-frame sustained proximity, 0.5 m minimum separation, exponential decay, five-second carry limit, reset after more than two seconds without a visible ball, and period-boundary reset.
3. Replace the brittle all-or-nothing kinematic conjunction with a tunable touch-confidence rule. Use ball kink, deceleration, turn, proximity, and team separation as evidence while retaining proximity and ambiguity protections.
4. Make angle and movement thresholds configurable for tuning against simulated data.

### Validation requirements

Report precision and recall together: touches per minute, in-play frames with a touch state, signed team assignment accuracy, false touches per minute, short-possession team accuracy, and leave-one-match-out paired results. The goal is higher touch coverage without losing the existing ambiguity, decay, carry, missing-ball, and period-reset safeguards.

## Summary Roadmap Table

| Step | Phase Target | Primary File | Regression Risk | Expected Output |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Noise Simulator Calibration | `possession/noise/simulate.py` | **Zero** (opt-in arg) | Closes simulation-to-reality gap (true 0.59+) |
| **2** | Asymmetric Restart Penalty | `possession/decode.py` | **Low** (default fallback) | Increases short-possession count post-stoppage |
| **3** | Targeted 10-Frame Context | `possession/features.py` | **Low** (appends columns) | Captures micro-surges without spatial flicker |
| **4** | Kinematic Impulse (±6f Kink) & Closing Velocity | `possession/features.py` | **Low** (isolated column) | Detects instant touch events at turnover frames |
| **5** | Soft Boundary & Weak Tagging Priors | `possession/features.py`, `emission.py` | **Low** (soft feature / prior) | Guides dead-ball calls without parallax kill switches |
| **6** | Signed Touch State & Possession Persistence | `possession/features.py` | **Low** (guarded additive) | Covers full possession span; $< 2\text{s}$ accuracy 50% → 59% |
| **7** | Touch Detector Recall & Team Ownership | `possession/features.py` | **Low** (guarded thresholds) | Improve signed-touch coverage while retaining ambiguity and carry safeguards |

---

## Historical Agent Execution and Logging Protocol

When this roadmap was active, each step followed this workflow:
1. **Implement:** Write clean, vectorized NumPy code with non-breaking defaults.
2. **Verify:** Run test suites and validation scripts (e.g., `pytest`).
3. **Log in `CHANGES.md`:** Immediately append a formal entry to [`CHANGES.md`](CHANGES.md) detailing:
   - Step number and title
   - File path modified
   - Exact line ranges (`L<start>-L<end>`)
   - Role and mechanism of the changes
   - Test output and regression verification confirmation.
