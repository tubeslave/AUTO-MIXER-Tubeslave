# ADR: macOS local and public-release evidence profiles

Status: superseded on 2026-07-15 by
`m121-macos-release-evidence-hardening-20260715.md`

The M121 decision removes the npm `mac:release` success surface. The supported
public-release lane is now the fixed system-shell launcher documented by M121;
the `mac:release` command below is retained only as historical context.

## 1. Context

The repository previously had one Electron packaging path whose output could be
mistaken for a public release even when it used development signing or no
notarization. The packaged app also needed an explicit source-to-package chain:
the PyInstaller backend, Vite output, Electron main/preload files, runtime
configuration, signatures, entitlements, archives, offline runtime behavior and
post-cleanup evidence must all refer to the same clean source tree.

Electron 28.3.3 was outside the supported stable lines in July 2026. The runtime
is therefore pinned to Electron 43.1.0. The packaging migration deliberately
keeps electron-builder 24.13.3 and `@electron/osx-sign` 1.0.5 pinned so the
runtime upgrade and build-tool upgrade are not coupled in one change.

## 2. Options considered

1. Keep one packaging command and describe unsigned/development artifacts in
   documentation.
2. Treat any locally signed arm64 app as release-ready.
3. Split local validation from public release, fail the release profile closed,
   and retain machine-verifiable evidence that can be checked again after the
   temporary build workspace is deleted.
4. Replace the Electron desktop application during this audit.

## 3. Decision

Choose option 3.

- `mac:local` produces an arm64 validation ZIP and an explicit
  `releaseEligible: false` manifest. It uses a verified ad-hoc signature by
  default; an explicitly configured local identity is optional. Neither form
  implies notarization or public distribution readiness.
- `mac:release` requires a clean checkout, exact Node/npm/Python/pip,
  Developer ID Application identity, Apple Team ID and keychain notary profile.
- macOS builds use collision-resistant run directories and an atomic build lock
  because release `npm ci` and local packaging share `frontend/node_modules`.
- Release child processes receive an allowlisted environment. Alternate Apple
  credential families and build-injection variables fail validation.
- The signed app contains an attestation for source HEAD/tree plus isolated
  backend and frontend hashes. Verification additionally compares the packaged
  Electron main/security/preload sources and every shipped runtime config file
  byte-for-byte with the clean checkout.
- App, helper, framework and backend entitlements are exact allowlists. The app,
  backend, ZIP application and DMG application must be arm64, Developer ID
  signed, timestamped and mode-preserving copies of one verified app tree.
- The packaged runtime smoke is continuously monitored under an immutable
  offline lock. The packaged backend process tree is additionally confined by
  a canonical macOS sandbox profile that denies non-loopback outbound traffic
  and passes TCP/UDP, IPv4/IPv6 and descendant-process probes. The smoke must
  reject an invalid token, authenticate a fresh token,
  expose analyze-only state, hide hardware controls, block the exact connection
  operations, open no external/WING socket, create no temporary write evidence,
  and leave the canonical manual-write log, decision traces, decision memory,
  console WAL, backend bundle, package files and app tree unchanged. Backend
  readiness must be measured against an exact `20000 ms` budget.
- Initial verification checks the retained isolated outputs. After `.build` is
  removed, the same verifier must pass again using the initial proof, signed
  attestation, build metadata and packaged-input comparisons. Only the second
  pass writes `release-manifest.json`.

## 4. Why this won

The split makes a useful local package possible without weakening the meaning
of a public release. The final manifest is derived from signatures, hashes,
Gatekeeper/notary results and runtime observations rather than from the build
command's exit code. Post-cleanup verification prevents an apparently valid
release from depending on temporary files that no longer exist.

## 5. Rejected alternatives

- Documentation-only labels: rejected because an artifact can outlive its
  console output and be redistributed without the warning.
- Development signature as release proof: rejected because it does not prove
  Developer ID distribution, Apple notarization, stapling or Gatekeeper.
- Keeping the complete `.build` workspace forever: rejected because it retains
  a large dependency environment and still does not independently bind every
  packaged runtime input.
- Upgrading Electron, electron-builder and the signing library together:
  rejected for this checkpoint because it obscures compatibility failures and
  enlarges rollback scope.
- Claiming public release readiness from source tests or a local smoke alone:
  rejected because Apple and clean-checkout evidence are external gates.

## 6. Implementation plan

1. Keep local and release builder configurations separate.
2. Pin the runtime and build/signing tools used by the release path.
3. Build backend/frontend into unique isolated directories and record locks,
   tool versions and output identities.
4. Sign an in-app build attestation and verify all packaged source/config inputs.
5. Run the continuous offline/no-WING packaged smoke.
6. Notarize and staple the DMG, validate app/DMG/ZIP signatures and exact
   entitlements, then write the initial verification record.
7. Delete `.build`, repeat verification from retained evidence and only then
   write the final release manifest.

The Python requirements and PEP 517 build prerequisite are exact-version and
SHA-256 locked for the reviewed arm64/CPython 3.14.3 artifact set. Installation
uses isolated pip, `--require-hashes`, no cache and no build isolation. Of 131
runtime/build distributions, 130 use wheels. `watchdog==6.0.0` has no compatible
upstream wheel and remains a hash-bound sdist; its compiled wheel bytes are not
bit-for-bit reproducible across repeated builds. This is an explicit residual
reproducibility limit, not an unbound input.

## 7. Test plan

- Unit/contract tests for profile validation, environment rejection, atomic
  build serialization, source drift and retained-evidence validation.
- Mutation tests for smoke product state, WING/network observations, source/app
  binding, entitlements, notarization schema and archive replacement.
- Full frontend tests and TypeScript compilation after the Electron upgrade.
- A current-tree ad-hoc-signed arm64 local build followed by the complete
  packaged offline smoke. The verifier must inspect every Mach-O file in the
  app and backend trees, not only the main executables.
- A clean-checkout Developer ID build that performs both verifier passes,
  notarization/stapling and Gatekeeper assessment before public release.

## 8. Risks and rollback

Electron 43 may expose runtime or packaging incompatibilities with the pinned
older builder/signing stack; the current-tree local package and smoke are the
required compatibility gate. Exact entitlement checks may fail when Electron
changes helper requirements, in which case the checked-in allowlist must be
reviewed rather than relaxed automatically. macOS currently marks
`sandbox-exec` deprecated; the offline smoke fails closed if its canonical probe
matrix stops working, but a supported replacement must be selected before the
mechanism disappears. The hash-bound `watchdog` source build means a Developer
ID run must not yet be described as bit-for-bit reproducible from locks alone.

Rollback is a source revert of the profile/evidence changes and Electron pin.
It may restore local development packaging, but it must not relabel an old
artifact as public-release eligible or delete historical smoke/notary evidence.

## 2026-07-12 implementation evidence

The local build at
`frontend/dist/local/20260712T055849Z_18b5ae9e-ec35-47eb-9eb7-77b78d12473b_local/`
binds dirty source content by SHA-256, carries a matching embedded local
attestation and passes deep/strict ad-hoc signature verification. Independent
inspection found `360` app Mach-O files and `345` backend Mach-O files, all
arm64. The extracted ZIP app tree matches the packaged app tree exactly.

The bound runtime smoke at
`artifacts/packaged_runtime_smoke/20260712T060214Z_bb83311f-1043-4f35-9ff5-6933708e0abe_packaged_runtime_smoke/result.json`
passed with backend readiness `4917/20000 ms`, zero external/WING sockets and
all four canonical stores unchanged. This is local compatibility evidence
only: dirty checkout, ad-hoc signature, `releaseEligible=false`, no Developer
ID, notarization, stapling or Gatekeeper claim.

## Final identity-bound package evidence

The superseding local build is
`frontend/dist/local/20260712T072932Z_9e3d9f29-5ccf-41ca-a46e-f70933a45b23_local/`.
At PyInstaller build time it embeds a self-hashed manifest for the 15
safety-critical source files. Frozen startup fails closed if that manifest is
missing, malformed or changed. Challenged runtime identity binds the embedded
manifest, exact backend executable, fresh memory-only Electron launcher-token
digest and the backend PID observed in the launched process group.

The superseding smoke result is
`artifacts/packaged_runtime_smoke/20260712T073254Z_af976647-7612-419d-a5a5-9922930b203e_packaged_runtime_smoke/result.json`.
All 18 identity checks passed; backend readiness was `5038/20000 ms`; all
`360` app and `345` backend Mach-O files were arm64; no external or WING socket
opened; and all four canonical stores, the backend bundle, package files and
app tree remained unchanged. The macOS package contract suite passed `34`
tests. The build is still dirty-tree-bound, ad-hoc signed and
`releaseEligible=false`; it is not Developer ID, notarization, stapling,
Gatekeeper or public-release evidence.
