# M117 Operator Diagnostic Truth Contract

## Task

Keep the current M116 operator-attended WING diagnostic truth internally
consistent while M110-HW01 remains blocked on `master_channel_signal_missing`.

## Evidence

- The latest `RUN_LEDGER.jsonl` row is an explicitly operator-approved,
  operator-attended diagnostic. It truthfully records
  `hardware_contacted=true`, `contacts_backend=true`, `contacts_wing=true`,
  `live_writes=false`, and no live authority.
- `tests/test_wing_mvp_rails.py` already accepts that narrow diagnostic branch
  and pins its two evidence hashes plus the non-machine-verified observation
  class.
- `Docs/CURRENT_STATE.md` still says every latest externally blocked ledger row
  must keep hardware contact false. That historical absolute rule now
  contradicts the ledger and the current test contract.
- `Docs/ACTIVE_MILESTONE.md`, `Docs/ROADMAP.md` and
  `ops/autopilot/BLOCKERS.md` describe the 3-second and 120-second captures as
  direct facts without consistently saying they were operator-observed and
  not independently machine-verified.

## Required outcome

1. Replace the stale absolute no-contact wording with a typed authority rule:
   ordinary unattended/control-plane rows remain no-contact; the exact
   operator-approved M116 diagnostic may record read-only contact while all
   write/readiness/authority flags remain false.
2. Mark every current M116 capture statement as operator-observed and not
   independently machine-verified.
3. Add a regression rail that prevents the stale absolute wording and requires
   the current operator-observation qualifier on the canonical truth surfaces.
4. Preserve strict live readiness at 88/100, normal allowlist 0, reset 0,
   corrections 0, live writes false, and standing authority false.
5. Do not contact backend, CoreAudio or WING. Do not change production code or
   live policy.

## Verification

- Focused M116/current-truth rails.
- Entire `tests/test_wing_mvp_rails.py`.
- Standard full backend suite.
- JSON/JSONL validation, memory lint, `git diff --check`.
- Independent read-only patch review.
