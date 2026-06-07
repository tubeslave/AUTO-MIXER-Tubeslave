# Codex Proposal: WING Read-Only Preflight Decision Artifact Hash Binding - 2026-06-07

## Short thesis

Operator review decisions should pin the exact `summary.json` and
`events.jsonl` bytes they reviewed, not only the artifact paths.

## What I understood

The WING read-only preflight gate is still waiting for real hardware evidence.
The safe offline work left before that step is to make the review and decision
path stricter so future physical evidence cannot be accidentally accepted after
being replaced or edited.

## Proposed solution

Compute SHA256 digests for file-backed preflight `summary.json` and
`events.jsonl` artifacts inside `backend/wing_readonly_preflight_evidence.py`.
Expose those digests through artifact review rows and pending decision
templates.

Extend `backend/wing_readonly_preflight_review_decision.py` so any decision for
a candidate with recorded digests must include matching `summary_sha256` and
`events_sha256`. Missing or mismatched digests fail closed as replacement
required.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- focused WING preflight evidence/review/decision/no-WING tests
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-decision-artifact-hash-binding-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_ARTIFACT_HASH_BINDING_20260607.md`

## Alternatives considered

- Keep path-only binding. Rejected because a path can point to different bytes
  after replacement.
- Add free-form operator notes with hashes. Rejected because the ledger needs
  deterministic machine validation.

## Risks

Existing pending review templates must be regenerated because they do not carry
the new digest fields. This is acceptable because the first read-only WING
blocker is still open.

## Test plan

- Compile changed Python modules.
- Run focused WING preflight evidence/review/decision/no-WING tests.
- Run quick no-WING release check, direction guard, memory lint, diff check,
  and full pytest before commit.

## Where the other agent may disagree

Kimi may consider path plus run id sufficient. It is weaker than byte-level
binding because the operator decision is a durable record and should survive
artifact movement or replacement audits.
