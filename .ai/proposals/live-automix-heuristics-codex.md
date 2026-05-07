# Live automix heuristics - Codex proposal

## 1. Short thesis

Implement the high-confidence live automix heuristics as deterministic,
testable control-plane modules first. Keep high-risk or research-heavy DSP as
documented future work.

## 2. What I understood

The project already contains modules named gain sharing, auto gate, auto pan,
auto EQ, compression, and phase alignment. The biggest gap is that some modules
are heuristic placeholders or static processors, not the exact live automixing
rules listed in the review.

## 3. Proposed solution

Add a Dugan-style automix controller that computes gain from each channel's
level relative to total active power, then applies NOM depth, Last Hold, and
optional gain limiting. Wire this into AutoFader V2 behind a new
`controller_mode: "dugan"` so existing gain-sharing and dynamic modes remain
available.

Update gate settings to safer live defaults and add an instrument preset helper
for drums, vocals, and sustained instruments. Extend AutoPanner with a
lightweight band accumulator so same-band sources are distributed evenly and
physical channel order keeps lower-numbered inputs nearer the center.

## 4. Likely files to touch

- `backend/auto_fader_v2/balance/dugan_automixer.py`
- `backend/auto_fader_v2/controller.py`
- `backend/auto_gate_caig.py`
- `backend/auto_panner.py`
- `config/default_config.json`
- `tests/test_dugan_automixer.py`
- `tests/test_gate.py`
- `tests/test_auto_panner.py`
- `Docs/adr/live-automix-heuristics.md`

## 5. Alternatives considered

- Replace existing gain sharing in place. Rejected because the current LUFS
  balance logic may still be useful and has different musical behavior.
- Implement dynamic EQ and spectral phase now. Rejected because both are larger
  latency/safety projects that need more validation and UI observability.

## 6. Risks

- Fader automation can create gain surprises if applied as absolute values.
  Mitigation: preserve relative command path and fader ceiling.
- Dugan attenuation may feel too aggressive for music. Mitigation: configurable
  Auto Mix Depth and full-gain limit.
- Gate timing changes can affect existing tests. Mitigation: update tests to
  assert the documented live-sound ranges.

## 7. Test plan

- Unit-test equal-level NOM behavior: 1 mic = 0 dB, 2 mics = -3 dB, 4 mics =
  -6 dB when full-gain limit is disabled.
- Unit-test Last Hold and Auto Mix Depth.
- Unit-test gain limiting behavior.
- Unit-test gate presets and panner same-band distribution.
- Run focused tests, then the standard project pytest command if feasible.

## 8. Where the other agent may disagree

Kimi may argue for delaying integration until a full signal-flow refactor exists.
My view is that a separately selectable Dugan mode is safer because it adds the
missing mature heuristic without forcing a full live-pipeline rewrite.
