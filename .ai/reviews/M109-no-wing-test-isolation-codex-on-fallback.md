# Codex critique of the independent M109 proposal

## 1. Strongest part

The proposal cleanly separates two truths: canonical invalid artifacts must
continue to fail closed, while static constant-contract tests must not inherit
mutable hardware history. Its opt-in helper is smaller and safer than a broad
fixture.

## 2. Possible mistakes or blind spots

The proposal correctly leaves nearby negative tests alone, but their results
can still contain unrelated canonical-artifact violations. That is test-noise,
not a demonstrated regression; record it as deferred hermeticity debt instead
of expanding this patch.

The proposal should also explicitly prove the real canonical aggregate still
fails for the expected single reason. A read-only assertion in the validation
evidence is enough; changing or freezing the hardware artifact store is not.

## 3. Where the solution is too complex, fragile, or costly

It is not materially over-complex. Adding a production injection seam, moving
artifacts, or creating an autouse fixture would all increase fragility.

## 4. What should be kept

- Test-only implementation.
- Explicit empty roots and payloads.
- Patch the `release_check` module global.
- Use only in the three positive constant-contract tests.
- Re-run explicit fail-closed coverage and the complete suite.

## 5. Revised recommendation

Adopt the independent proposal. Add one test-local helper, repair exactly three
tests, and record before/after evidence that the default canonical aggregate
still reports `wing_readonly_preflight_invalid_artifacts_present` with no
production diff.
