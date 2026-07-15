# M121 CP02 patch review

## Verdict

`CLEAN` for the offline implementation. No remaining P0, P1 or P2 finding was
found in the final snapshot. This review does not claim that Apple accepted a
real submission; the first Developer ID release run remains external evidence.

Final reviewed snapshot SHA-256:

```text
49519067f2bb5cafe5bd689f7197c9cf4dd86f8e455a9e5cd19f815b259a3500
```

The digest covers the binary tracked diff plus the SHA-256 identities of the
five new CP02 source/test files.

## Findings closed during review

1. Artifact identities recorded with relative paths were initially resolved
   against the process working directory. They now resolve against the release
   artifact root and have a real `artifactFileIdentity()` regression test.
2. `--allow-legacy` could initially reach retained/publication semantics. It is
   now strictly inspection-only and cannot publish any release manifest.
3. The DMG accepted ticket was initially not tied to the final DMG signature.
   The v2 evidence now binds the unique top-level ticket, SHA-256 and unchanged
   pre/post-staple CDHash.
4. App ZIP creation initially exposed a pathname replacement window. The ZIP
   is now created through a reserved no-follow descriptor in a private `0700`
   directory.
5. App state was initially captured after native metadata reads and app
   stapling still depended on a mutable pathname. State is captured first and
   stapler runs from a verified `dev/ino/realpath` application working inode.
6. DMG stapling initially depended on the mutable snapshot pathname. Both
   staple and validation now use the held `/dev/fd/3` descriptor.
7. Application ticket matching initially allowed an arbitrary prefix through
   suffix comparison. It now requires the exact submitted archive/app path.
8. Apple submission wait had no explicit deadline. It is bounded at 30 minutes.
9. Final review found that legacy inspection still required the three new v2
   sidecars. The final implementation accepts only the complete pre-v2 shape,
   rejects a partial legacy/v2 mixture, and keeps v2 sidecars mandatory for the
   normal release lane.

## Final validation

- Focused release/notarization contracts: `65/65 passed` before the final full
  run; focused legacy verifier re-review: `26/26 passed`.
- Full frontend suite after the P2 repair: `220/220 passed`.
- Production Vite build: passed, `58 modules transformed`.
- `node --check` for every changed release/notarization script: passed.
- `package.json` parse and `git diff --check`: passed.
- Independent correctness/security re-review: `CLEAN`.
- WING, OSC and live backend were not contacted.

The standard repository test command was also exercised in a clean worktree
before this review. It stopped at `1 failed, 2279 passed, 1 skipped` on
`test_real_historical_artifact_and_owner_decision_match_immutable_hashes`
because the exact ignored 2026-07-11 M005 physical artifact is absent from a
clean checkout. The same isolated test passes in the main checkout containing
those immutable bytes. This is a pre-existing evidence-portability blocker for
the next checkpoint, not a CP02 regression.

## Residual external evidence

A real clean-checkout release must still prove Developer ID signing, accepted
app and DMG Apple logs, stapling, Gatekeeper acceptance and retained-pass
verification. CP02 only makes that evidence fail-closed and retainable; it does
not synthesize or claim it.
