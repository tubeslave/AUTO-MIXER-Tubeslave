# M120 — HW01 teardown readiness-invalidation repair

## Goal

Make the proof-owned route-only runner report the real accepted HW01 result
without weakening disconnect, readiness or clean-teardown safety.

## Checkpoints

1. `M120-CP00`: preserve the real reproduction, Council proposals/critiques
   and ADR.
2. `M120-CP01`: add the teardown-only receiver and adversarial tests.
3. `M120-CP02`: pass focused M110, route-proof, CP05 and WING rails.
4. `M120-CP03`: pass full pytest, no-WING validation and independent review.
5. `M120-CP04`: commit source/truth, then create one marker-only CP05 rebind.
6. `M120-HW01`: repeat route-only proof from clean source and require accepted
   artifact plus verified capture stop, mixer disconnect, route revocation and
   clean backend termination.

## Safety invariants

- Normal live allowlist remains empty.
- No software-development run contacts WING after the source diff begins.
- No accepted route artifact can bypass teardown verification.
- Readiness invalidation remains fatal outside explicit disconnect teardown.
- No console parameter writes are added or executed by M120.

## Required validation

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_attended_soundcheck_runner.py -q
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_master_capture_route_proof.py tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
git diff --check
```

## Rollback

Revert the M120 implementation commit and create a new marker-only CP05 rebind.
Do not use the previously accepted route artifact as live authority; it expired
with its proof-owned process and remains evidence of the bug reproduction only.
