# Independent proposal: M121 release entrypoint and notary evidence

Status: superseded in its npm-entrypoint section by the accepted ADR after a
real `script-shell` zero-execution false success was reproduced. The notary
direction remains the proposal input.

## 1. Short thesis

Do not weaken the existing release environment validator. Put a pinned npm
lifecycle validator in front of it, and take explicit ownership of application
notarization so every successful Apple decision has a retained log identity.

## 2. What I understood

npm 10.9.8 injects eleven ordinary lower-case `npm_config_*` values into the
documented script. The core validator consequently rejects a legitimate run.
Separately, electron-builder/@electron/notarize consumes the successful app
submission ID internally and returns no evidence, while the DMG script only
downloads a log when Apple rejects the submission.

## 3. Proposed solution

- Add an npm-only wrapper that accepts exactly the pinned lifecycle surface,
  validates lifecycle, engine, user-agent and path semantics, rejects unknown
  or upper-case configuration keys, then scrubs every npm metadata key before
  entering the unchanged core builder.
- Set release `mac.notarize=false` and use an `afterSign` hook. The hook runs an
  explicit app notary client before ZIP/DMG creation.
- Share submission/log validation between app and DMG: success still downloads
  the JSON log, whose job ID, accepted status and archive SHA must match the
  exact private submitted bytes.
- Retain app/DMG log sidecars and evidence documents; require all four files in
  both release-verifier passes and bind them into manifest v2.

## 4. Likely files to touch

The files named in the Codex proposal, with focused tests in
`macosBuildProfilesContract.test.mjs`, `macosSigningContract.test.mjs` and
`macosReleaseVerifier.test.mjs`.

## 5. Alternatives considered

- Relax the core `npm_config_*` ban: rejected.
- Parse electron-builder debug output or query latest notary history: rejected
  because neither provides an unambiguous, supported submission identity.
- Keep successful logs only in terminal output: rejected because the final
  artifact would outlive the evidence.

## 6. Risks

The app hook must execute after Developer ID signing and before archive targets.
Temporary archives and evidence outputs need exclusive/no-follow creation,
stable identities and cleanup. Apple log fields must be treated as required
contract inputs rather than loosely parsed diagnostics.

## 7. Test plan

Exercise exact npm lifecycle acceptance plus injection failures; accepted and
rejected Apple responses; log UUID/status/SHA mismatches; pre-existing and
symlink outputs; app/DMG mutation races; missing or swapped four-file evidence;
and initial-to-retained manifest identity drift.

## 8. Where the other agent may disagree

The main trade-off is implementation size. A DMG-only patch is smaller, but it
does not close the application submission evidence gap that triggered M121.
