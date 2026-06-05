# Offline No-WING Validation Runbook

Use this runbook when the real mixer is not connected. The purpose is to prove
operator workflow, safety gating, proposal shape, simulated readback, rollback,
and reporting without touching live hardware.

## Boundary

Allowed:

- offline fixtures;
- Product Layer read models;
- proposal previews;
- simulated apply/readback/rollback;
- emergency-stop block proof;
- frontend/backend tests;
- package/build smoke.

Forbidden:

- real WING connection;
- OSC/MIDI live writes;
- direct `set_fader`, `set_gain`, EQ, routing, snapshot, FX, training, or rule
  writes;
- claiming autonomous/live readiness from offline evidence.

## Commands

One-shot release-style no-WING check:

```bash
python3 tools/no_wing_release_check.py \
  --include-package \
  --artifact-root artifacts/no_wing_release_check/current
```

Fast backend/offline preflight:

```bash
python3 tools/no_wing_release_check.py \
  --quick \
  --skip-frontend \
  --artifact-root artifacts/no_wing_release_check/current
```

Backend safety baseline:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Focused offline validation tests:

```bash
PYTHONPATH=backend python -m pytest tests/test_offline_validation_harness.py -q
```

Run the offline proposal harness:

```bash
python3 tools/offline_operator_validation_harness.py \
  --artifact-root artifacts/offline_validation/current_no_wing
```

Frontend checks:

```bash
npm --prefix frontend run test:e2e
npm --prefix frontend test -- --watch=false
npm --prefix frontend run build
```

Whitespace/diff check:

```bash
git diff --check
```

## Success Criteria

The no-WING validation can be treated as passed only if:

- harness status is `passed` or `clear`;
- `tools/no_wing_release_check.py` status is `passed` for release-style handoff;
- `connection_policy` is `no_hardware_no_network_no_live_writes`;
- `live_write` is `false`;
- `read_surface_fixture_evidence` is `passed`;
- `read_surface_fixture_evidence_schema` is `passed`;
- `gain_dry_run_proof` is `passed`;
- `supervised_proof_report_gain_dry_run_schema` is `passed`;
- `autonomy_evidence_overlay_schema` is `passed`;
- `next_wing_proof_plan_schema` is `passed`;
- `supervised_proof_report_gap_rollup_schema` is `passed`;
- `summary.json` and `summary.md` expose the Supervised Gain Dry-Run Proof as
  read-only, non-live, non-promoting evidence for fake-mixer gain pre-read,
  apply readback, rollback, cooldown block, and emergency-stop block behavior;
- `summary.json` and `summary.md` expose the Autonomy Evidence Overlay as
  read-only, non-live, non-promoting target-level proof evidence with no
  automatic safety update and `full_autonomous_ready=false`;
- `summary.json` and `summary.md` expose the Next WING Proof Plan as
  read-only and operator-required with `execution_allowed_by_plan=false`;
- `summary.json` and `summary.md` expose Read Surface Fixture Evidence as
  read-only, non-live, non-autonomous evidence, including the remaining physical
  readback fixture gaps;
- `summary.json` and `summary.md` expose the read-only Autonomy Gap Rollup with
  Full Autonomous still not ready and Supervised live-write kinds matching the
  current operator-mode policy;
- Analyze, Assist, and Auto have zero simulated apply count;
- Supervised proposals have simulated readback and rollback proof;
- unsupported/upward changes are blocked in tests;
- artifacts are written under `artifacts/offline_validation/`;
- release-check artifacts are written under `artifacts/no_wing_release_check/`;
- frontend tests still prove unsafe writes are not emitted.

## Readiness Wording

Correct wording:

- `offline dry-run validated`;
- `gain dry-run proof captured`;
- `proposal workflow simulated`;
- `supervised path remains the only live write candidate`;
- `real WING validation still required for hardware proof`.

Incorrect wording:

- `autonomous ready`;
- `fully live ready`;
- `EQ/routing/snapshot apply approved`;
- `offline simulation proves real hardware behavior`.

## macOS Package Note

The local no-WING package smoke uses the Electron mac build with timestamping
disabled in `frontend/package.json`. This avoids a false failure when the
signing timestamp service is unavailable. Public distribution/notarization
still requires a separate signed/notarized release check.
