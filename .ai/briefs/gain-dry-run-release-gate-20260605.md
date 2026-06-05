# Brief: gain-dry-run-release-gate-20260605

## Context

Full Autonomous remains blocked, and Supervised AutoMix is currently limited to
the existing fader/gain policy boundary. The nearest next promotion target is
`gain_trim_supervised_promotion`, but the current no-WING release bundle did
not run the existing fake-mixer gain proof runner or require the supervised
proof report to consume that artifact.

## Goal

Make every no-WING release check include a local supervised gain dry-run proof
that exercises the Product Layer proposal path, verified readback, rollback,
cooldown blocking, and emergency-stop blocking without connecting to WING.

## Constraints

- Do not widen live write authority.
- Do not run physical WING proof.
- Keep direct `set_gain`/trim/preamp command checks forbidden in release-check
  command lists.
- Treat gain dry-run evidence as proof planning only, not promotion authority.

## Acceptance

- `tools/no_wing_release_check.py --quick --skip-frontend` runs the gain dry-run
  proof before building the supervised proof report.
- The proof report schema check fails if the gain dry-run row is missing or
  claims promotion/autonomous readiness.
- The no-WING summary exposes the next physical WING proof plan as read-only,
  operator-required, and non-executing.
- Summary JSON/Markdown expose the gain dry-run evidence as read-only,
  non-live, non-promoting evidence.
