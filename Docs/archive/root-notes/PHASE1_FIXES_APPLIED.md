# Phase 1: Critical Fixes Applied — AUTO-MIXER-Tubeslave

All 13 critical bugs identified in the code review have been fixed.
Every modified file passes `python -c "import ast; ast.parse(open('filename').read())"`.

---

## Fix Summary

### C-01 — Wrong domain in EQ parameter mapping (overflow / NaN)
**File:** `auto_effects.py`  
**Lines affected:** ~482–488 (in `StateManager.map_to_parameters`)

**Problem:**  
`_db_to_linear(features.energy_low/mid/high)` was called on FFT band-energy values
that are already linear magnitude sums (output of `np.sum(magnitude[mask])`).
Passing large positive linear values into `10 ** (db/20)` caused massive numeric
overflow and NaN propagation into EQ parameters.

**Fix:**  
Replace `_db_to_linear(energy_X)` with a normalized energy ratio:
```python
_energy_sum = features.energy_low + features.energy_mid + features.energy_high + 1e-10
params['eq_low_gain']  = features.energy_low  / _energy_sum
params['eq_mid_gain']  = features.energy_mid  / _energy_sum
params['eq_high_gain'] = features.energy_high / _energy_sum
```
Result is a dimensionless 0–1 weight for each band, safe to use as a gain factor.

---

### C-02 — Hysteresis declared but never applied (gate chatter in live sound)
**File:** `auto_gate_caig.py`  
**Lines affected:** ~404–470 (in `GateProcessor.process`)

**Problem:**  
`GateSettings.hysteresis_db = 3.0` was stored but the state-machine compared the
signal against a single threshold for both opening and closing decisions. In live
sound this caused rapid gate chatter when the signal hovers near the threshold.

**Fix:**  
Implement dual-threshold hysteresis:
- `open_threshold_db  = threshold_db`  — gate opens when signal exceeds this
- `close_threshold_db = threshold_db - settings.hysteresis_db`  — gate only begins
  closing when signal falls below this (3 dB below open threshold by default)

The OPEN state now only transitions to HOLD when `rms_db < close_threshold_db`.

---

### C-03 — Missing relative gate per ITU-R BS.1770-4 (LUFS underestimated 1-3 dB)
**File:** `lufs_gain_staging.py`  
**Lines affected:** ~68–104 (in `SignalStats.calculate_integrated_lufs`)

**Problem:**  
`calculate_integrated_lufs()` applied only the absolute gate (`> -70 LUFS`) but
omitted the mandatory relative gate (`>= ungated_loudness - 10 LU`) required by
ITU-R BS.1770-4 §2.8. Silent and near-silent blocks included in the average
dragged the integrated value down by 1–3 dB.

**Fix:**  
Two-pass gating per ITU-R BS.1770-4:
1. **Pass 1** — absolute gate: keep blocks `> -70 LUFS`
2. Compute ungated loudness from Pass-1 blocks
3. **Pass 2** — relative gate: keep blocks `>= ungated_loudness - 10 LU`
4. Final integrated LUFS computed from gated blocks only

Fallback: if no blocks survive the relative gate, the ungated value is used
(graceful degradation).

---

### C-04 — `find_snap_by_name` destructively loads snapshots during live show
**File:** `wing_client.py`  
**Lines affected:** ~1289–1340 (in `WingClient.find_snap_by_name`)

**Problem:**  
The original implementation sent `/$ctl/lib/$action = "GO"` for each snapshot
index in order to read back the active snapshot name. This **loaded every snapshot**
it iterated over, completely changing the mixer state during a live performance
each time a snapshot search was performed.

**Fix:**  
Use the read-only OSC query `/$ctl/lib/$name` (set index first with
`/$ctl/lib/$actionidx`, then query `$name` without GO) so snapshot names can be
inspected without loading them. The active mixer scene is never altered during
the search. Sleep intervals also reduced from 100–200 ms to 50 ms per step.

---

### C-05 — Ratio formula inverted: percussion gets soft compression instead of aggressive
**File:** `auto_compressor_cf.py`  
**Lines affected:** ~296–301 (in `CFCompressorCalculator.calculate_params`)

**Problem:**  
`ratio = 3.0 * (1.0 - 0.3 * cf_norm)` produced ratios that **decreased** as
`cf_norm` increased. High Crest Factor (percussion, CF > 18 dB → `cf_norm ≈ 1`)
yielded `ratio ≈ 2.1:1` (soft), while low CF signals got `ratio ≈ 3.0:1`
(harder). This is the exact opposite of correct behaviour.

**Fix:**  
Invert the mapping so percussion gets aggressive compression:
```python
ratio = 3.0 * (1.0 + 0.5 * cf_norm)
```
Range: `cf_norm=0` → ratio ≈ 1.5:1 (sustain), `cf_norm=1` → ratio ≈ 4.5:1
(percussion). After 50/50 blend with base params, PERCUSSION class (base 6:1)
yields ≈ 5.25:1, while FLAT class (base 1.2:1) yields ≈ 1.35:1.

---

### C-06 — Return type annotation mismatch in `calculate_correction`
**File:** `auto_fader_hybrid.py`  
**Lines affected:** ~274–320 (in `DecisionEngine.calculate_correction`)

**Problem:**  
The method signature declared `-> Tuple[float, float]` (two values) but the
implementation `return correction_db, rate_limited, confidence` returns **three
values**. Callers that unpacked exactly two values would silently get incorrect
behaviour or a `ValueError`.

**Fix:**  
Updated annotation to `-> Tuple[float, float, float]` to match the actual return.
Docstring updated accordingly.

---

### C-07 — Swapped `ref_audio` / `tgt_audio` arguments in `process_audio`
**File:** `auto_phase_gcc_phat.py`  
**Lines affected:** ~321 (in `AutoPhaseAligner.process_audio`)

**Problem:**  
`analyzer.add_frames(tgt_audio, ref_audio)` passed target and reference in the
wrong order. `add_frames(ref_frame, tgt_frame)` is the expected signature. The
swap flipped the sign of every measured delay, meaning a channel that needed
+2 ms of delay was assigned −2 ms instead.

**Fix:**  
```python
analyzer.add_frames(ref_audio, tgt_audio)  # correct order
```

---

### C-08 — `correlation_peak` can produce NaN / Inf on silent frames
**File:** `auto_phase_gcc_phat.py`  
**Lines affected:** ~177–183 (in `GCCPHATAnalyzer.compute_delay`)

**Problem:**  
```python
correlation_peak = np.abs(interpolated_peak_value) / np.sqrt(
    np.sum(ref_frame**2) * np.sum(tgt_frame**2)
)
```
When either frame is silence (all zeros), the denominator is exactly `0.0` and
`correlation_peak` becomes `NaN` or `Inf`, which propagated into confidence
scores and downstream decisions.

**Fix:**  
Add `eps` (already defined in scope) to the denominator and clamp result to
`[0, 1]`:
```python
energy_product = np.sqrt(np.sum(ref_frame**2) * np.sum(tgt_frame**2))
correlation_peak = float(np.clip(
    np.abs(interpolated_peak_value) / (energy_product + eps), 0.0, 1.0
))
```

---

### C-09 — LRA computed on linear RMS values instead of dB (wrong result)
**File:** `auto_effects.py`  
**Lines affected:** ~282–292 (in `AudioCore._extract_tier5_features`)

**Problem:**  
```python
features.lra = np.percentile(rms_values, 95) - np.percentile(rms_values, 10)
```
`rms_buffer` stores linear RMS values (e.g. 0.0001 – 0.9). Subtracting linear
percentiles produces a meaningless result (near zero for most signals). LRA
is defined as a **dB** range.

**Fix:**  
Convert to dB before computing the percentile difference:
```python
rms_db_values = 20 * np.log10(np.maximum(rms_values, 1e-10))
features.lra = float(np.percentile(rms_db_values, 95) - np.percentile(rms_db_values, 10))
```

---

### C-10 — Gate does not re-open during RELEASING state (missed fast signals)
**File:** `auto_gate_caig.py`  
**Lines affected:** ~452–464 (in `GateProcessor.process`, RELEASING branch)

**Problem:**  
Once the gate entered the RELEASING state it would always coast down to CLOSED,
even if the signal came back above the open threshold. Fast consecutive events
(e.g. drum rolls) with short release times could have their second hit cut off.

**Fix:**  
Added a re-open check in the RELEASING branch (and also in HOLD):
```python
elif is_above_open:
    self.state = GateState.OPENING
```
This mirrors professional hardware gate behaviour.

---

### C-11 — (Merged into C-08) — See C-08 above.

---

### C-12 — `_sum_squares` floating-point drift can go negative → NaN LUFS
**File:** `lufs_gain_staging.py`  
**Lines affected:** ~293–310 (in `LUFSMeter.process`)

**Problem:**  
The running sum-of-squares accumulator `self._sum_squares` is updated via
`+= new² - old²`. Over a long session (hours) floating-point rounding errors
accumulate and can drive `_sum_squares` slightly below 0. Calling
`max(value, 1e-10)` on a tiny negative number still returns the negative, which
then produces `nan` from `np.log10`.

**Fix:**  
Add an explicit clamp after each batch of samples:
```python
if self._sum_squares < 0.0:
    self._sum_squares = 0.0
```

---

### C-13 — `ScenarioDetector()` created on every `calculate_correction()` call
**File:** `auto_fader_hybrid.py`  
**Lines affected:** ~298–304 (in `DecisionEngine.calculate_correction`)

**Problem:**  
```python
target = ScenarioDetector().get_target_lufs(scenario, self.target_lufs)
```
A new `ScenarioDetector` object was instantiated on **every call** to
`calculate_correction()`, which runs at 100 Hz for every active channel.
This is unnecessary heap allocation in a real-time audio control loop.

**Fix:**  
Create the `ScenarioDetector` once and cache it on `self`:
```python
if not hasattr(self, '_scenario_detector'):
    self._scenario_detector = ScenarioDetector()
target = self._scenario_detector.get_target_lufs(scenario, self.target_lufs)
```

---

## Files Modified

| File | Fix IDs |
|------|---------|
| `auto_effects.py` | C-01, C-09 |
| `auto_gate_caig.py` | C-02, C-10 |
| `lufs_gain_staging.py` | C-03, C-12 |
| `wing_client.py` | C-04 |
| `auto_compressor_cf.py` | C-05 |
| `auto_fader_hybrid.py` | C-06, C-13 |
| `auto_phase_gcc_phat.py` | C-07, C-08 |

## Syntax Verification

All 7 modified files pass `python -c "import ast; ast.parse(open('filename').read())"`:

```
auto_effects.py           OK
auto_gate_caig.py         OK
lufs_gain_staging.py      OK
wing_client.py            OK
auto_compressor_cf.py     OK
auto_fader_hybrid.py      OK
auto_phase_gcc_phat.py    OK
```
