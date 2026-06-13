# WING MVP Handoff Prestart Summary Command Binding

## Summary

The WING read-only handoff validator now treats the validated plan artifact as
the source of truth for the no-WING prestart `summary.json` path. Preview and
capture commands must both pass that path with `--prestart-summary`.

The no-WING handoff summary must show:

- `plan_backend_port_prestart_summary_path`
- `preview_command_prestart_summary`
- `preview_command_prestart_summary_matches_plan: true`
- `capture_command_prestart_summary`
- `capture_command_prestart_summary_matches_plan: true`

This is no-WING handoff hardening. It does not contact WING, start a backend,
write mixer state, grant live authority, or close the WING read-only connection
proof blocker.

## Validation

Completed before commit:

- Touched Python compile check: passed.
- Focused no-WING/Project Director/WING rail tests: `264 passed`.
- no-WING quick release check with backend port `65535`: passed.
- Generated no-WING handoff summary showed
  `preview_command_prestart_summary_matches_plan: true` and
  `capture_command_prestart_summary_matches_plan: true`.
- Project Director stayed on `WING read-only connection proof` with
  `guard_status=on_target`, `guard_score=90`, and
  `guard_stop_the_line=False`.
- Direction guard stayed `on_target`, `score=90`, and
  `stop_the_line=False`.
- Full pytest: `3289 passed, 1 skipped, 1 warning`.
