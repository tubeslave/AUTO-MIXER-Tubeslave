# Runtime artifact path isolation

## Context

The `show_assist_auto_v1` correction event path was relative to the current
working directory. A backend launched from `backend/` appended to the tracked
`backend/artifacts/show_assist_auto_v1/gated_correction_events.jsonl`; tests
launched from the repository root created a second untracked source-tree file.
Runtime execution therefore dirtied the checkout it was meant to verify.

## Options considered

1. Keep cwd-relative paths and ignore the generated files.
2. Make the configured path absolute inside the repository.
3. Resolve relative correction artifacts below a canonical per-user runtime
   root, with a pytest-only isolated override and traversal rejection.

## Decision

Use option 3. Production writes resolve below
`~/.automixer/runtime/`; pytest writes use
`AUTOMIXER_TEST_CORRECTION_RUNTIME_DIR` (or the existing isolated console
runtime fallback) only while `PYTEST_CURRENT_TEST` is present. Absolute paths
and `..` traversal are rejected.

## Why this won

The path no longer depends on launch cwd and cannot escape into tracked source.
Tests remain hermetic, while historical files stay untouched for provenance.

## Rejected alternatives

Ignoring cwd-relative output was rejected because tracked files remain mutable
and clean-checkout evidence would still drift. An absolute repository path was
rejected because it formalizes the contamination instead of removing it.

## Implementation plan

1. Add a strict runtime path resolver to the correction executor.
2. Resolve once at executor construction and append only to the resolved path.
3. Change the default config to a runtime-relative profile path.
4. Isolate the path in the shared pytest fixture and ignore historical runtime
   directories without deleting or rewriting them.

## Test plan

- Run correction-executor and AutoSoundcheck tests from a changed cwd.
- Assert events appear only under the per-test runtime root.
- Assert absolute paths, parent traversal and non-pytest test overrides fail.
- Confirm both historical source-tree artifacts retain their prior size, mtime
  and SHA-256 after tests.

## Risks and rollback

Operators looking for the old cwd-relative file must use the new per-user path.
The historical files remain available as evidence. Rollback is a source revert,
but must not resume writes into the checkout.
