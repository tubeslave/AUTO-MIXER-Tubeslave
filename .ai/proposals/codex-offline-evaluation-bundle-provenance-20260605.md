# Proposal: offline-evaluation-bundle-provenance-20260605

## Short thesis

Add a small offline bundle validator so ABX, safety audit, render checksum, and
gate-report evidence can be reviewed as one provenance unit.

## What I understood

Automixer needs stronger offline evaluation evidence before any model/DSP
candidate can be considered useful. The existing ABX gate checks loudness,
peaks, checksums, preference, and attached safety audits. The next safe gap is
not live behavior; it is proving that the separate files used in evaluation
belong to the same candidate render.

## Proposed solution

Create `tools/offline_evaluation_bundle.py`. It reads a local JSON manifest,
computes file checksums, loads JSON references, and verifies:

- candidate render checksum is present and matches the file;
- safety audit passes and targets that candidate checksum;
- ABX artifact B/candidate checksum matches the candidate render;
- ABX artifact carries attached safety-audit provenance for that render;
- ABX gate report passed, is offline-only, has `live_write=false`, and names the
  ABX artifact checksum it evaluated;
- tool metadata is present.

## Likely files to touch

- `tools/offline_evaluation_bundle.py`
- `tests/test_offline_evaluation_bundle.py`
- `.ai/briefs/offline-evaluation-bundle-provenance-20260605.md`
- `.ai/proposals/codex-offline-evaluation-bundle-provenance-20260605.md`
- `Docs/adr/offline-evaluation-bundle-provenance-20260605.md`

## Alternatives considered

- Extend `tools/offline_abx_gate.py` again. Rejected because the ABX gate should
  judge one artifact, while the bundle validator judges file-level custody
  between artifacts.
- Generate real audio fixtures in this cycle. Rejected because the useful safety
  step is the manifest contract; real render generation can come later.
- Treat a passing ABX gate report as enough. Rejected because stale reports can
  be copied unless they name the exact ABX artifact checksum.

## Risks

This is provenance validation, not audio-quality proof. It does not prove a
candidate is good, only that the evidence files are tied together and offline.

## Test plan

```bash
python3 -m py_compile tools/offline_evaluation_bundle.py tests/test_offline_evaluation_bundle.py
PYTHONPATH=backend python3 -m pytest tests/test_offline_evaluation_bundle.py -q
python3 tools/memory/memory_lint.py
/opt/homebrew/bin/git diff --check
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

## Where the other agent may disagree

Kimi may prefer extending the ABX gate output schema first. That is reasonable,
but a separate bundle validator lets existing and future gate reports be tied to
their source artifacts without turning the ABX gate into a broader custody tool.
