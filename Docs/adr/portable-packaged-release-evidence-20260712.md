# ADR: Portable packaged release evidence paths

## Context

The macOS build profile, build metadata, notarization evidence, packaged-runtime
smoke bundle and final release manifest recorded absolute workstation paths.
Hashes still detected byte changes, but a copied evidence bundle could not be
verified at a new location and leaked the builder's local home path. Replacing
those paths with unchecked relative strings would create a different problem:
a crafted manifest could escape its artifact root or follow a symlink to an
unrelated file.

## Options considered

1. Keep absolute paths and treat evidence as machine-local.
2. Rebase any missing absolute path by filename during verification.
3. Record artifact-root-relative paths and resolve them through a fail-closed
   path policy.

## Decision

Use explicit `artifact_root_relative_v1`,
`packaged_build_artifact_root_relative_v1`,
`manifest_directory_relative_v1`, and
`artifact_root_relative_and_logical_v1` path policies for all new release
evidence. Packaged app, attestation, archive, backend bundle, package-file,
notarization and smoke sidecar identities are recorded relative to their
declared artifact root. Source locks, toolchain inputs and extracted-ASAR
comparisons use normalized logical identities. External safety stores use
logical `repository_root` or `operator_home` identities instead of
workstation-specific absolute paths.

The detached `verify-packaged-runtime-smoke.mjs` verifier resolves sidecars
relative to the manifest directory, requires the canonical `result.json` and `result.md`
siblings, checks their size and SHA-256, and cross-checks the result's safety
and build-profile fields. Traversal, absolute paths under a relative policy,
and symlink escape are rejected.

Release verification preflights the top-level DMG, ZIP and notarization
evidence before reading their bytes or invoking codesign, stapler, hdiutil or
ditto. Each must be an existing regular file (never a symlink or special file),
and its real path must remain inside the release artifact root. The
notarization-evidence binding repeats that regular-file and realpath check.
The notarization command itself requires the release artifact root and runs
the same top-level regular-file/containment gate before codesign, notarytool or
stapler. The build orchestrator also rejects a symlink DMG before launching the
notarizer.

At verification completion the top-level artifact set is discovered and
validated again. Artifact identities open files with no-follow semantics, hash
the opened descriptor twice, and require stable inode, size, mode, link count,
mtime and ctime across both passes. The pathname must still name that exact
regular-file object. Identical-byte symlink replacement and same-inode
mutate/restore races therefore cannot be promoted.

Stable JSON and identity reads run in small child helpers whose working
directory pins the already validated parent directory object. The helpers
verify root, parent, realpath, device and inode identities before and after two
reads. Replacing an intermediate parent with a symlink between containment and
open therefore fails instead of redirecting the read to a stable external
file.

New evidence files are created by a tiny helper whose working directory pins
the already validated parent directory object. It uses exclusive no-follow
creation, verifies canonical root/parent identities after fsync, and removes
the exact created inode on failure. DMG replacement uses a pinned-root helper
with a hard-link rollback of the original file.

Final manifests serialize immutable identities captured by the last complete
verification pass; they never reopen mutable artifact paths while building the
JSON. The eligible filename is published only after source, application-tree
and retained-artifact postconditions pass: a hidden exclusive candidate is
hard-linked atomically to the final name and then removed. The build
orchestrator reads initial/final manifests and build profiles through the same
stable no-follow JSON reader and removes the exact final manifest if a later
source guard fails.

The candidate writer returns the SHA-256/size of the exact intended JSON and
the publisher requires that identity before the hard-link publication. Source,
application tree and every retained artifact are revalidated again after the
eligible filename exists; failure removes only the exact published object.

Native `codesign`, `stapler`, Gatekeeper, `ditto` and `hdiutil` checks operate
on private app/DMG/ZIP snapshots, never on the mutable canonical inputs. Each
snapshot is bound to the canonical bytes/tree before use and revalidated after
the native command. Notarization additionally binds the working DMG after
codesign and submit, then binds the stapled identity across validation before
an inode-checked replacement of the canonical DMG.

Missing `pathPolicy` is invalid by default. Legacy manifests are accepted only
through an explicit `allowLegacy` API option or CLI `--allow-legacy`. Under that
opt-in, every path the verifier dereferences must still resolve inside the
current artifact root. Opaque legacy cross-root labels are never used to open a
file by the detached verifier and must pass the packaged-artifact containment
check in release verification. A legacy path is never silently rebased after
relocation. Deleting `pathPolicy` therefore cannot silently downgrade a current
manifest.

## Why this won

The evidence can move as a bundle while retaining exact byte bindings. The
resolver has one auditable containment rule and cannot be redirected to an
arbitrary same-named file outside the bundle.

## Rejected alternatives

Machine-local absolute paths were rejected because they are not portable.
Filename fallback was rejected because it can verify the wrong artifact after
relocation. Unchecked relative paths were rejected because `..` and symlinks
can escape the evidence root.

## Implementation plan

- Add shared relative identity and safe resolution helpers.
- Emit relative paths from macOS build profiles and packaged smoke evidence.
- Emit portable build metadata, notarization evidence, initial verification and
  final retained release manifests.
- Verify smoke sidecars independently before release verification consumes
  them.
- Require explicit legacy opt-in and preserve only containment-safe legacy
  behavior.

## Test plan

- Verify a build profile after copying its complete artifact root.
- Verify a detached smoke bundle after relocation.
- Assert serialized build metadata and final release manifests contain no
  builder, repository, app or temporary absolute paths.
- Reject traversal, external absolute paths, and symlink escape.
- Reject top-level DMG, ZIP and notarization-evidence symlinks before any
  downstream release command can run.
- Reject parent-directory swap, same-inode mutate/restore, build-profile
  symlink and post-verification artifact replacement races.
- Reject candidate-byte substitution, post-publication retained-artifact drift,
  working-DMG substitution and intermediate-parent redirection.
- Reject a missing path policy by default and accept legacy fixtures only with
  explicit opt-in.
- Run the focused macOS build-profile and release-verifier Node tests.

## Risks and rollback

Consumers that assumed absolute `path` values must resolve them against the
declared root. The macOS CI workflow uses the shared build-profile resolver for
the packaged app, ZIP and embedded attestation instead of treating recorded
relative paths as workflow-CWD paths. Rollback is to remove the new path
policies and helpers together;
keeping relative emission without the containment-aware verifier is unsafe.
This decision changes evidence addressing only and grants no live mixer
authority.

The remaining local trust boundary is a concurrently malicious process owned
by the same macOS account that can discover a randomized private path, replace
an object only while a native Apple tool has it open, and restore the exact
bytes and metadata before the postcondition read. Path-only Apple command APIs
do not accept our already-open descriptor, so public release evidence must be
created in a clean isolated CI/operator account. This residual is not treated
as release eligibility and grants no WING authority.
