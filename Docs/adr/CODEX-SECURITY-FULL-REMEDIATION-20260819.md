# ADR: Full Codex Security remediation ledger and fail-closed cohorts

## 1. Context

The sealed partial Deep Scan `2c00402a-b02e-4b80-a68f-c9c079eb70c4`
reported 143 first-party findings: 77 high, 59 medium, and 7 low. Existing local
remediation is substantial but does not explicitly disposition every original
occurrence. The collaborative checkout is heavily dirty, the Codex Security
connector is unavailable in this task, and this project controls live sound.

## 2. Options considered

1. Apply generated workbench patches mechanically.
2. Fix occurrences one by one in severity order.
3. Freeze a complete ledger, then remediate severity tiers in file-clustered
   cohorts using a small shared quarantine guard only for unsupported legacy
   mutators and sink-local checks for supported paths.
4. Move work into a clean-room branch or introduce a new capability broker.

## 3. Decision

Choose option 3.

Create a schema-checked 143-row ledger before production edits. Every row must
have either current disposition evidence or an assigned executable proof task.
Remediate high, medium, then low findings, clustering by file within each tier.
Unsupported legacy live mutation remains dry-run/plan-capable but fails closed
before transport through a small shared quarantine guard. Supported paths keep
operation-specific sink-local peer, authority, value, pre-state, readback,
rollback, and result checks.

Local success is `locally_verified_unsealed`; only a fresh full current-checkout
Codex Security scan can close findings in the workbench.

## 4. Why this won

The complete ledger prevents silent omissions. File clustering minimizes
conflicts with unrelated dirty work. A narrow shared quarantine guard makes
unsupported legacy mutators auditable without creating another authorization
framework, while sink-local checks retain the stronger guarantees required for
supported cyber-physical paths.

## 5. Rejected alternatives

- Mechanical workbench apply is unsafe against overlapping dirty files and has
  already been refused by workbench state persistence.
- Pure occurrence ordering repeatedly edits central files and makes rollback
  unclear.
- A clean-room branch would misbind or lose the current collaborative state.
- A new capability broker is too broad, adds a new critical trust boundary, and
  is unnecessary for quarantine.
- Reusable confirmation phrases are not operation-specific authority and do
  not provide physical readback or rollback.

## 6. Implementation plan

1. Freeze the sealed occurrence ledger and validate 143 rows plus 77/59/7.
2. Re-run currently cited remediation tests and assign proof tasks to every
   unexplained row.
3. Reproduce any reported validation-harness blocker before changing tests.
4. Fix high findings in file-clustered cohorts.
5. Fix medium and low findings using the same contracts.
6. Validate every cohort, then reconcile the ledger and final report.
7. Run a fresh full Security scan when the connector is available.

The ledger checker is data-only. It may validate schema, completeness, and a
code-owned mapping of stable test node IDs; it must never execute shell commands
stored in ledger data.

## 7. Test plan

- Exact ledger count and severity histogram.
- Pre-fix or recorded base-revision exploit proof, post-fix negative proof, and
  positive bounded behavior for each root-cause cohort.
- Owning-module tests and syntax/diff hygiene per slice.
- No-WING quick per cohort with all authority denied.
- Standard full pytest at severity milestones and finalization; expected
  CP05/lease dirty-tree blockers are reported separately and never deselected.
- Frontend, dependency, native, and Node checks where touched.
- Fresh full current-checkout Security scan for final closure.

## 8. Risks and rollback

Shared quarantine code is a common dependency, so keep it small and test it in
isolation. Central files may contain unrelated hunks; snapshot scoped diffs and
rollback only remediation-owned hunks. Low disk space requires bounded temp
roots and no broad repository copies. Any failed positive control returns the
affected ledger row to blocked. Software tests never imply WING/dLive movement,
acoustic safety, or concert readiness.
