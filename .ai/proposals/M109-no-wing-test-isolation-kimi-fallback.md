# Independent proposal — M109 no-WING contract-test isolation

## Council degradation note

Kimi CLI was invoked in read-only plan mode on 2026-07-13. It returned HTTP
402: `We're unable to verify your membership benefits at this time.` No Kimi
proposal was produced and no files were changed. A fresh read-only Council
agent supplied the independent proposal below.

## 1. Short thesis

Use test-only empty-evidence injection. No production change is justified: the
production validator correctly fails closed when canonical evidence contains
invalid artifacts, while only three constant-contract tests accidentally
depend on mutable repository history.

## 2. What I understood

The three tests assert static proof-runner, run-summary, and allowed-value-type
agreement. The validator nevertheless calls the default evidence builder,
which recursively reads canonical artifacts and deliberately rejects any
invalid record. Existing tests already demonstrate how to build and inject an
empty evidence snapshot; separate tests preserve invalid-artifact and authority
fail-closed coverage.

## 3. Proposed solution

Add an opt-in private test helper that:

1. builds evidence with `artifact_payloads=()` and `artifact_roots=()`;
2. monkeypatches `release_check.build_wing_readonly_preflight_evidence`;
3. is called only by the three named constant-contract tests.

Both empty arguments are explicit: the first denies inline evidence and the
second prevents canonical filesystem discovery. Do not make the fixture
`autouse` and do not add dependency injection to the production validator.

## 4. Likely files to touch

- `tests/test_no_wing_release_check.py`
- Council records and ADR only

## 5. Alternatives considered

- Production dependency injection: rejected as unnecessary API expansion.
- Latest-valid-only production selection: rejected as a safety-rail weakening.
- Delete or relocate failed hardware artifacts: rejected because provenance is
  valuable and immutable.
- Patch constants or working directory: rejected as indirect and fragile.
- Autouse fixture: rejected because it could hide artifact-sensitive behavior.

## 6. Risks

- Reuse of the helper beyond these three tests could conceal artifact
  regressions.
- Patching the wrong namespace would leave the validator connected to the real
  builder; patch the module global used by the validator.
- Nearby negative tests retain extra real-artifact violations, but still assert
  their targeted mismatch; broad cleanup is not required by M109.

## 7. Test plan

1. Three repaired tests together.
2. Explicit invalid-artifact and authority fail-closed tests.
3. Complete `tests/test_no_wing_release_check.py`.
4. Standard full suite.
5. Diff/status audit proving no production or artifact changes.

## 8. Where Codex may disagree

Codex may prefer a reusable fixture or production injection point. The explicit
test-local helper is safer because its scope is visible and cannot silently
affect artifact-intake tests.
