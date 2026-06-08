# Brief: chatgpt-pulse-20260608

## Short thesis

Accept only the safe part of the 2026-06-08 ChatGPT Pulse snapshot: latency and
OSC measurement ideas are useful for Automixer only as separate lab/read-only
evidence around the existing WING proof rail, not as a new live-control PoC.

## What I understood

Pulse repeated several Automixer-relevant themes:

- OSC RTT and control-to-audio latency should be measured with p50/p95/p99
  evidence.
- Cross-track unmasking, phase correlation, PEQ/Diff-DEQ, and tiny ML heads are
  useful correction-method ideas, but they must remain offline/proposal-only
  before the supervised WING MVP is proven.
- Telemetry, rollback, signed diffs, and deterministic cycle reports matter,
  but this repo already has local trace, rollback, no-WING, and report gates.

Current project state already has:

- `tools/latency/osc_rtt_benchmark.py` and `tools/latency/audio_e2e.py` for
  no-WING latency smoke evidence.
- `Docs/MVP_WING_SUPERVISED.md` and `.ai/project-layer/wing_mvp_blockers.md`
  naming the first blocker as real WING read-only preflight.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` as the next hardware runbook.
- Dirty WING read-only preflight files from the current active worktree, so the
  Pulse slice must avoid those files unless it directly protects the runbook.

## Proposed solution

Do not add new OSC, audio-trigger, CamillaDSP, EasyEffects, JUCE/LV2/AU,
OPA/WASM, cosign, ONNX, CoreML, or Demucs tooling in this Pulse cycle.

Record one small runbook guard instead:

- WING read-only preflight must remain strictly read-only and physical-readback
  focused.
- Any latency/OSC/audio benchmark inspired by Pulse must be a separate lab-only
  artifact after the preflight artifact is captured and reviewed.
- The benchmark must not send tone-trigger, write, proposal, rollback,
  emergency-stop, connect, disconnect, or mixer-mutation requests as part of the
  read-only preflight proof.

## Likely files to touch

- `.ai/briefs/chatgpt-pulse-20260608.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.paperclip/reports/chatgpt_pulse_review_20260608T*.md`

`Docs/PROJECT_IMPLEMENTATION_PLAN.md` should not change because this is a
boundary clarification inside the current WING MVP rail, not a roadmap change.

## Alternatives considered

- Add a new WING OSC/audio E2E PoC from Pulse. Rejected because it can require
  hardware, tone triggers, and live transport behavior outside the current
  read-only proof.
- Add native DSP, AU/LV2/JUCE, or plugin-message-bus scaffolding. Deferred as
  broad lab/plugin architecture.
- Add OPA/WASM/cosign deterministic cycle signing. Deferred because it adds new
  tooling/dependency decisions and does not close the first WING blocker.
- Add ONNX/CoreML/Demucs separation work. Deferred because the project already
  has an offline stem-training loop with a deterministic mock splitter and
  heavier source-separation work needs a separate lab plan.

## Risks

- The useful latency idea may be forgotten if it is only documented. Mitigation:
  keep it explicitly tied to the existing latency tools and separate it from
  the WING read-only preflight proof.
- A future agent may treat latency smoke numbers as live readiness. Mitigation:
  the runbook guard says latency benchmarks are not readiness closure.
- The active worktree is already dirty in WING preflight code/tests, so this
  cycle must not modify those code paths.

## Test plan

- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

No Python module changes are planned for this Pulse slice.

## Where the other agent may disagree

Kimi may argue that Pulse Card 5 should directly extend the WING RTT tool. My
recommendation is to avoid that today: the current blocker waits for a real
read-only preflight artifact, and adding another benchmark path risks delaying
or confusing the first proof.
