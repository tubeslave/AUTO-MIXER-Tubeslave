# ADR: M121 macOS release entrypoint and notarization evidence v2

## 1. Context

The pinned npm 10.9.8 lifecycle injects eleven ordinary lower-case
`npm_config_*` keys, but the current core release validator rejects every npm
configuration key. Thus the documented package script fails before build.
Separately, successful application notarization is opaque inside
electron-builder/@electron/notarize, and the DMG script retains no successful
Apple log. A final release cannot prove the exact accepted submissions.

Kimi CLI was unavailable with membership error 402 before reading the brief.
Two independent read-only Codex reviews instead pressure-tested the design; the
primary agent remains the sole writer.

## 2. Options considered

1. Allow standard npm keys directly through the core release gate.
2. Keep direct Node only and leave the package script broken.
3. Add a narrow npm lifecycle wrapper while retaining direct Node as canonical.
4. Remove the npm release-success surface and clean the environment before Node.
5. Trust electron-builder success output and only improve DMG evidence.
6. Own app and DMG notarization, retain accepted logs, and verify both.

## 3. Decision

Choose options 4 and 6.

The package no longer defines `mac:release`. A configured npm `script-shell`
can return zero without executing any project code, so an npm exit code cannot
be release evidence. The only supported lane is
`/bin/sh -p scripts/run-macos-release`. The fixed system shell unsets Node
startup inputs, validates pinned Node/npm, and uses `/usr/bin/env -i` with an
explicit PATH and credential-name allowlist before Node starts. The inner `.mjs`
verifier requires that clean-environment handoff and then starts the unchanged
core gate without another shell.

The operator supplies the exact absolute pinned Node binary explicitly:

```bash
AUTOMIXER_NODE_BIN=/absolute/path/to/node-v22.22.3/bin/node \
  /bin/sh -p scripts/run-macos-release
```

electron-builder internal app notarization is disabled. An awaited `afterSign`
hook explicitly submits a private app ZIP before distributable archives are
built. App and DMG transactions always fetch accepted raw Apple logs and require
matching log format, UUID, Accepted/statusCode, archive filename, SHA-256 and
upload time. The verifier matches the Apple ticket's arm64 app-bundle code item
(`digestAlgorithm=SHA-256`) to identical final app-bundle and main-executable
codesign CDHashes, closing the submitted-to-final bridge. Four contained
sidecars feed a v2 portable two-pass manifest. The DMG transaction likewise
requires the unique top-level SHA-256 Apple ticket to match unchanged
pre/post-staple DMG CDHashes. App ZIP creation and both write-capable stapler
calls use pinned descriptors or a pinned working-directory inode so pathname
replacement cannot redirect native writes. Notary waits are bounded at 30
minutes. `--allow-legacy` is inspection-only: it cannot run the retained pass
or publish any release manifest. The inspection path accepts the complete
pre-v2 shape without the three new app/raw-log sidecars, while rejecting a
partial mixture of legacy and v2 sidecars.

## 4. Why this won

It removes a misleading success surface without weakening the existing
child-build gate or pretending npm can police actions that occur before project
code. The notary design replaces success-by-exit-code with submission-, byte-, ticket-
and final-artifact-bound evidence that survives build-workspace deletion.

## 5. Rejected alternatives

- Broad npm allowlist in the core: too easy to extend accidentally.
- npm convenience release command: rejected after `/usr/bin/true` script-shell
  reproduced a zero-execution exit code 0.
- Debug-output or latest-history correlation: ambiguous under concurrent Apple
  submissions and not a supported evidence API.
- DMG-only success logs: leaves the shipped app's accepted submission unknown.
- `stapler validate` alone: does not independently bind the submitted private
  ZIP to the final app signature.
- Retain partial/v1 evidence as v2: violates fail-closed release semantics.
- Treat `--allow-legacy` as release eligibility: crafted old evidence could be
  republished under the current schema.

## 6. Implementation plan

1. Land the clean-environment launcher and hostile npm/Node regression tests.
2. Add shared notary response/log validation.
3. Add explicit app hook and strengthen DMG flow.
4. Require four evidence files and ticket-cdhash binding in verifier v2.
5. Bind all identities across initial and retained passes.
6. Validate, commit, release lease, then perform one CP05 marker-only rebind.

## 7. Test plan

- Fixed-shell engine/path checks; `NODE_OPTIONS` preload cannot execute; direct
  `.mjs` entry is rejected; hostile npm script-shell cannot false-green a
  removed release script.
- App/DMG accepted, rejected, malformed, oversized and missing log responses;
  UUID/status/statusCode/SHA/name/time mutations.
- Wrong-app ticket, cdhash/path/arch mutations and swapped app/DMG logs.
- Root/parent/app symlink, rename/tree races, pre-existing output and output
  replacement attempts around every native stage, including descriptor-pinned
  ZIP creation and app/DMG stapling.
- Four-file stable identity across both verifier passes; final manifest v2 only
  after retained verification.

## 8. Risks and rollback

Apple may evolve optional log fields; required identity fields intentionally
fail closed and require reviewed schema change. Explicit app notarization changes
ownership but not order: electron-builder 24.13.3 awaits `afterSign` before
ZIP/DMG creation. Failed partial output remains in its unique run directory and
is never eligible. Rollback reverts the source commits and discards that run;
old evidence is preserved under its original schema and never promoted.
