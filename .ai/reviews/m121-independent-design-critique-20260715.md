# Independent critique: M121 release evidence design

## 1. Strongest part of the other proposal

The proposal preserves the default-deny core environment gate, replaces an
ambiguous app-notary black box with an awaited repository-owned `afterSign`
transaction, and keeps release eligibility false until the retained-artifact
pass. electron-builder 24.13.3 confirms the required order: sign application,
await `afterSign`, then build distributable targets.

## 2. Possible mistakes or blind spots

1. npm may honor `npm_config_script_shell` before the JavaScript wrapper begins.
   Therefore the npm wrapper is a convenience integrity check, not the
   authoritative security boundary. The canonical invocation needs an outer
   fixed system shell plus `env -i` before Node starts.
2. The wrapper must also bind `npm_execpath`, `npm_node_execpath`, package root,
   lifecycle script/event/command and exact pinned user-agent, then scrub every
   npm metadata key.
3. Apple log validation needs `logFormatVersion`, `jobId`, `statusCode`, archive
   filename and upload-time bounds in addition to status and SHA. Raw accepted
   log bytes must be retained.
4. Archive SHA alone does not bridge the submitted app ZIP to the final stapled
   app. The verifier must match the accepted ticket's arm64 main-executable
   cdhash to the final Developer ID application signature.
5. Root, parent and app inode/realpath/tree identity must be checked around each
   native command. All evidence outputs must be exclusive and non-replaceable.

## 3. Where the solution is too complex, fragile, or costly

A claimed secure `npm run` boundary is invalid because npm owns the shell before
the wrapper. A real `/usr/bin/true` script-shell returned success without
executing project code, so the package release script must be removed and direct
Node must be the only release lane.
Keeping an entire temporary app ZIP is unnecessary if the retained Apple ticket
is cryptographically matched to the final app's main-executable cdhash.

## 4. What should be kept from the other proposal

- Exact rejection of every inherited npm lifecycle surface.
- `mac.notarize=false` and an awaited top-level app hook.
- Shared app/DMG submission-log state machine.
- Four retained contained files and a v2 two-pass manifest.
- Fail-closed missing, swapped, symlink and mutation tests.

## 5. My revised recommendation

Implement two checkpoints. First, remove the false-green npm release surface
and land a fixed-shell clean-environment launcher. Second, implement notary
evidence v2 with raw logs, complete Apple metadata checks, run-window bounds,
ticket-cdhash bridging and pinned filesystem identity. Because either tracked
change invalidates CP05, finish M121 before one marker-only rebind and before
HW02-HW05.
