# M121 macOS release evidence hardening

## Problem

The strict 95/100 audit found two independent public-release P1 defects on the
current committed tree:

1. The documented `npm run mac:release` entrypoint is unusable. npm 10.9.8
   injects lower-case `npm_config_*` lifecycle variables, while
   `validateReleaseEnvironment()` intentionally rejects every such input.
2. Successful Apple notarization logs are not retained or identity-bound.
   electron-builder hides the app submission identifier on success, and the
   separate DMG notarizer only fetches a log on failure.

## Required outcome

- Keep the core release environment gate default-deny.
- Remove `npm run mac:release` as a release-success surface. A configured npm
  `script-shell` can return zero without executing project code, so the only
  authoritative entrypoint must be a fixed system shell that clears the
  environment before Node starts.
- Own app notarization explicitly so both app and DMG submissions retain the
  accepted Apple log and bind it to the exact submitted bytes and final stapled
  artifact.
- Make the two-pass release verifier fail closed on missing, swapped, escaped,
  stale or mutated notary evidence.
- Preserve `releaseEligible=false` until retained-artifact verification passes.
- Do not contact WING, change live authority, or claim strict readiness gain.

## Constraints

- Exact release toolchain remains Node 22.22.3, npm 10.9.8 and CPython 3.14.3.
- Only keychain-profile notary credentials are allowed; never log credentials.
- The source checkout must remain clean for an actual release run.
- All evidence paths must be artifact-root-relative regular files with no-follow
  creation and stable identity checks.
- The patch must be split into independently testable npm-entrypoint and notary
  evidence changes.

## Acceptance tests

- `/bin/sh -p scripts/run-macos-release` clears pre-execution Node/npm inputs
  through `env -i`, pins Node/npm and PATH, then launches an inner verifier and
  the unchanged core gate without another shell. A real hostile `script-shell`
  cannot false-green because the package has no `mac:release` script.
- App and DMG accepted flows always fetch and validate logs.
- Submission/log UUID, status and archive SHA mismatches fail.
- Evidence/log symlink, path escape, replacement and identity drift fail.
- Focused frontend contract tests, production frontend build and `git diff
  --check` pass before broader validation.

## Existing dirty state

The main checkout contains unrelated untracked generated audio/review artifacts.
They must remain untouched. The tracked tree was clean at M121 start, HEAD
`d5ac70a33ec4dd838a8d9284b04a8b9c992d34ea`.
