# Codex Proposal — M112 Current Validation Truth Sync

## 1. Short thesis

Land a finite fail-closed truth sync in two commits: all non-marker truth and
parser fixes first, then only the two CP05 marker files.

## 2. What I understood

The product/runtime baseline `451d0fdf3` is committed and validated, but the
canonical control plane still reports the older `5c054eb9e` validation and a
now-false M111 dirty-tail state. Hardware truth has not improved.

## 3. Proposed solution

- Record `451d0fdf3` as the validated product/runtime baseline, not as a claim
  about the future M112 commit identity.
- Synchronize the latest suite, rail and no-WING evidence across all canonical
  truth surfaces and append one zero-authority ledger row.
- Generalize suite-result regexes and derive the progress report's decision
  status and external requirement count from canonical data.
- Track exactly the three canonical no-WING capsule files.
- Commit all non-marker changes, recompute CP05 source binding, then commit only
  the marker pair.

## 4. Likely files to touch

`ops/autopilot/*`, `Docs/CURRENT_STATE.md`, `Docs/ACTIVE_MILESTONE.md`,
`Docs/ROADMAP.md`, the M001/M110/M112 plans, project-layer truth files,
`Docs/reports/MVP_PROGRESS_LATEST.md`, `tests/test_wing_mvp_rails.py`,
`tools/codex_autoloop/update_mvp_progress.py`, Council records, three capsule
files, and finally the two CP05 markers.

## 5. Alternatives considered

- A single commit cannot preserve the dynamic CP05 binding after non-marker
  changes.
- Writing the future commit SHA into its own contents creates a self-reference.
- Leaving the capsule untracked makes canonical truth depend on ephemeral local
  files.

## 6. Risks

Any tracked edit after marker rebind invalidates CP05. Generated evidence must
not be swept into Git. Historical launch-only provenance must not be rewritten.

## 7. Test plan

JSON/JSONL parsing, focused progress/rail tests, detached capsule verification,
WING/static rails, the standard full suite, memory lint, diff check, CP05
verification after both commits, and independent final review.

## 8. Where the other agent may disagree

Whether to fix the generator in M112 and whether the latest three-file capsule
belongs in Git. Both are necessary to prevent immediate truth regression.
