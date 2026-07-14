# M112 Current Validation Truth Sync

## Objective

Synchronize the canonical control plane with the already committed and tested
M111/M005 migration tree without contacting WING, opening audio/OSC transport,
or widening live authority.

## Confirmed baseline

- Validation-source commit: `451d0fdf398fe7f1f4ffd1c37f2e6f2d370506df`.
- Full suite: `7160 passed, 3 skipped in 416.18s`.
- WING/static truth rails: `150 passed`.
- No-WING capsule: `artifacts/no_wing_release_check/20260714T195349Z_no_wing_release_check`, 66/66 checks, detached manifest passed.
- Strict real-live readiness: `88/100`; normal allowlist `0`; standing authority `false`.
- M110-HW01 is blocked because the required physical WING 48x48 CoreAudio USB device is absent. Dante Virtual Soundcard is not a substitute.

## Scope

1. Correct stale validation facts in STATE, ledger, milestone, roadmap, reports,
   plans and project-layer blocker surfaces.
2. Make full-suite parsers accept an arbitrary non-negative skipped/warning
   count and remove two hard-coded progress-report facts.
3. Canonicalize only `summary.json`, `summary.md` and
   `artifact-manifest.json` from the latest no-WING capsule.
4. Commit the non-marker tree, then rebind CP05 in a separate marker-only
   commit.

## Hard boundaries

- No WING/backend/audio/network contact in this work item.
- No console writes, reset, soundcheck corrections or authority promotion.
- Keep the launch-only source commit at `5c054eb9e28d29601d974850b8874858b3fe0881`.
- Preserve unrelated generated/untracked evidence.
