# ChatGPT Pulse 20260604 Brief

## Short Thesis

Use the useful part of the Pulse snapshot as an offline evaluation gate:
deterministic A/B or ABX artifacts should prove seed/checksum evidence,
loudness matching, peak safety, and statistical preference before Automixer
claims an offline EQ/dynamics/render change is better.

## What I Understood

Pulse contained nine cards. The project already has safe slices for telemetry
learning data, streaming alerts, write-surface inventory, Diff-DEQ shadow
validation, and short autonomous Codex cycles. The strongest remaining idea is
a small ABX gate for offline evaluation artifacts. It does not require WING
hardware, live OSC writes, external tools, or new production dependencies.

## Proposed Solution

- Add a local offline ABX gate tool that reads JSON evaluation artifacts.
- Fail technical gates when seed/checksum evidence is absent, A/B LUFS differ
  by more than 0.5 LU, or either peak exceeds -0.1 dBFS.
- Compute exact two-sided binomial p-value and Wilson 95% interval for ABX
  choices.
- Allow technical artifacts to pass without a preference claim unless the caller
  explicitly requires preference evidence.

## Likely Files To Touch

- `tools/offline_abx_gate.py`
- `tests/test_offline_abx_gate.py`
- `.paperclip/reports/chatgpt_pulse_review_20260604T*.md`

`Docs/PROJECT_IMPLEMENTATION_PLAN.md` does not need a roadmap update for this
slice because the work fits the existing offline/shadow evaluation scope.

## Alternatives Considered

- Diff-DEQ preset exporters: useful later, but host-specific and more likely to
  become dependency or format churn.
- Rego/WASM policy benchmark: useful as infrastructure, but larger than one
  safe Pulse slice.
- Latency and OSC/DSP scripts: require hardware/network measurement to be
  meaningful.
- Telemetry schema: already represented by the repo telemetry layer.

## Risks

- ABX preference significance is not a substitute for operator judgment.
- LUFS and sample peak gates only validate comparison fairness and basic
  safety; they do not prove a mix is musically correct.
- Future audio rendering must still keep EQ/dynamics live writes blocked unless
  a separate supervised hardware proof exists.

## Test Plan

- `python3 -m py_compile tools/offline_abx_gate.py tests/test_offline_abx_gate.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_offline_abx_gate.py -q`
- `git diff --check`

## Where The Other Agent May Disagree

Kimi may argue that ABX tooling should be integrated into the offline replay
harness immediately. For this cycle, keeping it as a standalone tool is safer
because the worktree is dirty and the evaluation contract can be tested without
touching replay, Product Layer, or frontend files.
