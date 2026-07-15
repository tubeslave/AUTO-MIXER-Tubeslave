# ADR: M121 macOS release entrypoint and notarization evidence v2

## 1. Context

The pinned npm 10.9.8 lifecycle injects eleven ordinary lower-case
`npm_config_*` keys, but the current core release validator rejects every npm
configuration key. Thus the documented package script fails before build.
Separately, successful application notarization is opaque inside
electron-builder/@electron/notarize, and the DMG script retains no successful
Apple log. A final release cannot prove the exact accepted submissions. Even
after package evidence is retained, public readiness spans separate domains:
committed source and CP05, the notarized package, physical WING route/capture,
full HW02-HW05 closure, physical listening and proof-owned teardown. Mutable
runner paths and mismatched historical runs cannot form one immutable proof.

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
7. Treat package, CP05 and hardware artifacts as independent green checks.
8. Publish a separate immutable cross-domain bundle bound to one source.

## 3. Decision

Choose options 4, 6 and 8.

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

The cross-domain artifact uses
`automixer.public_release_evidence_bundle.v1`, separate from package manifest
v2. Its production builder requires the fixed clean Node 22.22.3 environment,
rejects preload and dynamic-loader variables, requires the selected repository
to be the repository executing the authority code, and compares every authority
file byte and executable mode to its exact HEAD blob at construction,
pre-publication and post-replay cuts. Helper processes execute the exact selected
Git-blob source through pinned Node input rather than a mutable helper pathname.
Hidden
`assume-unchanged`/`skip-worktree` index flags are rejected. Its retained Git
membership proof includes the commit object, the exact traversed tree-object
closure, the release authority closure, and the five source/CP05 blobs retained
for detached replay. The builder also recomputes the CP05
non-marker Git-tree binding, replays retained package evidence, snapshots the
mutable M110 summary/route plus the committed immutable M006 contract into a
fixed `0700`/`0600` private payload, and structurally validates only a claimed
full attended `show_mix_ready` execution with exact-source route proof,
recomputed canonical reset/M006 digests, exact receipt scope and operation
coverage, completed musical lanes,
fresh Main 47/48 evidence, matching
final audits, SHA-bound physical listening and verified teardown. The public
manifest is reconstructed as one exact sanitized projection and replayed before
publication; raw M110 bytes are private audit evidence. The M110 summary must
retain the runner's sorted compact Python JSON bytes with exactly one final
newline and no duplicate keys, alternative number spellings, or capability
material. The raw route is the backend's exact sorted compact Python JSON form
without a final newline. Route replay fixes the true-peak allowance at `+3 dB`, accepts
fractional acknowledgements in the summary's final UTC second, and replays the
exact hardware binding. A backend log is not
claimed until the runner can bind its exact post-teardown bytes.
Detached verification rehashes and revalidates structural consistency only.
The self-hash and Git object proof have no pinned public trust root and cannot
authenticate synthetic Apple/WING claims. The manifest therefore reports
`structural_consistency_verified`, `evidenceComplete=false`,
`externalAuthenticityVerified=false`,
`externalAuthenticatedAttestationRequired=true`, and
`readinessClosureAllowed=false`. Native-trust, Gatekeeper and physical claims
require a separately authenticated attestation/publication channel before they
can close public readiness.

Raw M110 bytes are captured through `O_NOFOLLOW` descriptors and the copied
JSON must match those descriptor-pinned size/SHA identities. The private
payload root remains open while a helper whose kernel-pinned working directory
is that exact root enumerates and hashes the exact eight flat `0600`,
single-link files; ancestor pathname replacement cannot redirect the walk.
The same pinned-cwd rule protects retained tree hashing. The build-profile
semantics must match the size/SHA identity of the exact bytes read, and raw
input closure is checked again after the final retained-package replay before
payload and public-manifest return cuts.

The production public-evidence launcher must run from `frontend/`. Raw M110
inputs must be outside the distributable release root:

```bash
cd /absolute/repository/frontend
AUTOMIXER_NODE_BIN=/absolute/node-v22.22.3/bin/node \
  /bin/sh -p scripts/run-public-release-evidence-bundle build \
  --repo-root /absolute/repository \
  --release-root /absolute/release-run \
  --m110-summary /absolute/private-inputs/execution-summary.json \
  --route-proof /absolute/private-inputs/route-proof.json

AUTOMIXER_NODE_BIN=/absolute/node-v22.22.3/bin/node \
  /bin/sh -p scripts/run-public-release-evidence-bundle verify \
  --release-root /absolute/release-run
```

## 4. Why this won

It removes a misleading success surface without weakening the existing
child-build gate or pretending npm can police actions that occur before project
code. The notary design replaces success-by-exit-code with submission-, byte-,
ticket- and final-artifact-bound evidence that survives build-workspace
deletion. The separate cross-domain schema prevents a valid package or
historical HW01 from being mistaken for a complete release while preserving
deterministically replayable private bytes behind a safe public projection.
It explicitly refuses to turn self-consistency into external authenticity.

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
- Extend package manifest v2 with hardware readiness: mixes independent trust
  domains and lets package verification appear to create live readiness.
- Publish raw M110 evidence directly: leaks LAN paths, channel state and nonces.
- Accept route-only HW01 or another source tree: does not prove HW02-HW05 or the
  shipped source.
- Treat the bundle self-hash or local Git object graph as a public trust anchor:
  both are self-authored and permit a fully synthetic but consistent fixture.
- Claim release readiness without a pinned signature/transparency identity:
  rejected; this patch records the missing external-authentication boundary.

## 6. Implementation plan

1. Land the clean-environment launcher and hostile npm/Node regression tests.
2. Add shared notary response/log validation.
3. Add explicit app hook and strengthen DMG flow.
4. Require four evidence files and ticket-cdhash binding in verifier v2.
5. Bind all identities across initial and retained passes.
6. Add a fail-closed cross-domain builder/verifier and private raw-evidence
   retention with a sanitized public projection.
7. Prove the real retained-v2 verifier success path, including the smoke
   manifest's build-profile size binding, then mutate a retained ZIP and require
   detached failure.
8. Pin every spawned evidence helper to its exact selected Git blob and reject
   transient pathname replacement.
9. Label detached output structural-only and forbid readiness closure until an
   external authenticated attestation exists.
10. Validate, commit, release lease, then perform one CP05 marker-only rebind.

## 7. Test plan

- Fixed-shell engine/path/cwd checks; the child has an exact environment,
  fixed `HOME`/temp/locale values and exact `PATH`; the launcher excludes
  `NODE_OPTIONS`. Every release-tool Git call uses a separate fixed environment,
  disables system/global config, fsmonitor, hooks, lazy fetch and replacement
  objects, pins the selected worktree, and first applies a repository-local
  config allowlist that rejects include/attribute/filter/diff/pager/hook command surfaces, so inherited
  `HOME`, `GIT_*`, local fsmonitor or clean-filter poisoning cannot execute
  during evidence reads. Command-line overrides force `core.filemode=true`,
  `core.ignorecase=false` and fixed attribute/exclude policy at each actual Git
  read, including mode-only and case-only drift checks. Direct Node CLI preload
  arguments make the ordinary release entry fail closed, but preload code runs
  before module entry and can mutate JavaScript process state; therefore a
  caller-selected hostile interpreter/preloader remains outside this local
  structural trust boundary and this tool is not a public trust anchor. The
  helper runtime pathname, realpath and inode/stat identity are pinned across
  every helper spawn; an unmarked direct `.mjs` entry is rejected under the
  trusted local runtime assumption; hostile npm script-shell cannot false-green
  a removed release script.
- App/DMG accepted, rejected, malformed, oversized and missing log responses;
  UUID/status/statusCode/SHA/name/time mutations.
- Wrong-app ticket, cdhash/path/arch mutations and swapped app/DMG logs.
- Root/parent/app symlink, rename/tree races, pre-existing output and output
  replacement attempts around every native stage, including descriptor-pinned
  ZIP creation and app/DMG stapling.
- Four-file stable identity across both verifier passes; final manifest v2 only
  after retained verification.
- Retained-package replay, private-payload identities and public-manifest bytes
  are rechecked in that order at standalone and builder return cuts; raw-input
  single-link closure is rechecked after the final retained replay.
- Stable-tree compatibility and inode/mode/symlink guards; canonical CP05 byte
  binding; full M110 source/route/M006/readiness/listening/teardown mutations;
  fixed-shell/direct-entry rejection; no app/backend/WING launch.
- Hidden Git index flags, authority byte/mode drift, unused Git tree objects,
  retained source mutation, duplicate summary keys, forged true-peak allowance,
  final-second fractional acknowledgements, noncanonical Python/route numbers,
  duplicate or reordered retained/public manifest keys, distributable-root raw
  M110 inputs, raw-input and copied-private-payload hardlink aliases, late
  hardlink creation, raw-input byte substitution, private-payload ancestor
  replacement, build-profile semantic-read substitution, hostile inherited-HOME Git config, repository-local
  fsmonitor/attribute/filter commands, transient helper pathname replacement,
  and post-publication authority drift.
- A complete synthetic retained-v2 package must pass the production detached
  structural replay while remaining explicitly ineligible to close readiness;
  changing the retained ZIP afterward must fail by exact identity.
- Runner M006 scope order must equal the server canonical prepared-write order
  for all three stages and match the retained golden hashes.
- A current historical HW01 or stale CP05 must fail cross-domain construction.

## 8. Risks and rollback

Apple may evolve optional log fields; required identity fields intentionally
fail closed and require reviewed schema change. Explicit app notarization changes
ownership but not order: electron-builder 24.13.3 awaits `afterSign` before
ZIP/DMG creation. Failed partial output remains in its unique run directory and
is never eligible. Rollback reverts the source commits and discards that run;
old evidence is preserved under its original schema and never promoted.
Raw attended evidence remains private. A failed bundle build leaves an
unpublished partial `0700` payload that must be discarded as a whole; paths are
never reused or overwritten. Rollback removes bundle tooling without relabeling
partial or historical evidence as authenticated or readiness-closing.
