# ADR: offline-evaluation-bundle-provenance-20260605

## Context

Offline render evaluation is becoming a safety boundary for future Automixer
DSP/model candidates. The ABX gate now checks loudness, peaks, checksum
coverage, preference evidence, and attached safety audits. The remaining gap is
bundle-level provenance: separate render, audit, ABX artifact, and gate-report
files can drift unless one manifest checks their checksums together.

## Options considered

1. Keep separate artifacts and rely on report text.
2. Add a strict offline evaluation bundle validator.
3. Fold file-custody checks into the ABX gate.

## Decision

Use option 2. Add `tools/offline_evaluation_bundle.py` as an offline-only JSON
manifest validator. It computes file checksums and verifies that the safety
audit, ABX artifact, and gate report all point back to the same candidate
render and source artifact.

## Why this won

The validator improves evidence quality without touching live transport or
runtime mixer behavior. It keeps the ABX gate focused on evaluation metrics and
adds a separate custody check for the bundle that reviewers can run locally.

## Rejected alternatives

Report text was rejected because it is easy to copy stale paths or checksums.
Folding everything into the ABX gate was rejected because bundle custody spans
multiple files and belongs in a separate offline/report pipeline tool.

## Implementation plan

- Add a pure Python bundle validator under `tools/`.
- Require manifest checksums for candidate render, safety audit, ABX artifact,
  and ABX gate report.
- Load JSON artifacts and check pass/offline/live policy fields.
- Add focused tests for complete provenance, audit mismatch, stale gate report,
  CLI output, and absence of forbidden live-write calls.

## Test plan

```bash
python3 -m py_compile tools/offline_evaluation_bundle.py tests/test_offline_evaluation_bundle.py
PYTHONPATH=backend python3 -m pytest tests/test_offline_evaluation_bundle.py -q
python3 tools/memory/memory_lint.py
/opt/homebrew/bin/git diff --check
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

## Risks and rollback

Risk: the validator can become another local artifact if nobody uses it in real
offline render runs. Rollback is deleting the validator, focused tests, and
Council artifacts. No live OSC/WING write path is added.
