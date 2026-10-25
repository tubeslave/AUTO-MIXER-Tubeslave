# Instrument-specific compression: first bounded task is bass

## User decision
Dmitry prefers the delivered Belye Stai v2 vocal: it neither jumps forward nor disappears. Record the scoped human acceptance against exact delivered hashes. Do not generalize 51/234 ms or vocal ratio to other instruments. Keep the failed intelligibility target as historical evidence; no machine safety gate or threshold is weakened.

## Goal and bounded implementation
Start other instruments with the bass. Expose its original first-compressor insert so the existing bass-role Compression Director v1.1 can replace it, not stack after processed audio. Reproduce the delivered bass on no-change. Source-bound controls explicitly freeze the second compressor's numerical threshold and the final static gain; its detector/envelope still respond to candidate audio. Audition level matching is separate and logged. Preserve all vocal/mastering/live code.

## Work done before PR
- Read current branch 5098d0a and its latest successful Tests and Stem Offline Test.
- Inspected saved original recipe, exact v2 runtime and source manifests.
- Added bass-only adapter and 17 new contract tests. New plus vocal adapter: 21 passed. New plus vocal/core/review-gate selection: 64 passed locally.
- Uploaded code blobs were verified identical to locally tested files by git object SHA.
- Full 207-second source preparation completed. No-change bass matches independently executed original procedural chain exactly, float PCM SHA a91affd0d0f39ee900f0f4ee71dc670cb92f3ea6b236f3ce48a3b3f2571f433b.
- Full session reproduction with the exact v2 vocal matches delivered v2 premaster SHA 73e407ac2893b94ea49aecfa4274da52c3131cc0dafb04fc394c0f94a596eb59; repeated graph is sample-identical.
- Three full-length bass candidates are being measured, not ranked. Existing critic snapshots are descriptive: a bass stability target is not silently substituted with vocal intelligibility or generic punch.

## Acceptance boundary
Code needs exact PR CI. Music needs level-matched listening; all new bass audio stays diagnostic. This is a frozen song adapter, not a completed universal bass director. No original WAV, delivered master, vocal settings, live control or neural model is changed.
