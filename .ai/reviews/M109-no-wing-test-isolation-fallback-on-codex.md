# Independent critique of the Codex M109 proposal

## 1. Strongest part

The Codex proposal draws the correct safety boundary: exactly three static
contract tests become hermetic, while production discovery, failed hardware
artifacts, and fail-closed validation remain untouched.

## 2. Possible mistakes or blind spots

- Build the empty evidence before monkeypatching. A patched lambda that calls
  `release_check.build_wing_readonly_preflight_evidence(...)` would recurse.
- The validation plan should explicitly run one invalid-evidence safety test,
  not rely only on the complete file.
- Do not freeze the repository's current `13/6/7` artifact counts as an
  integration contract; those counts are mutable evidence history.

## 3. Where the solution is too complex, fragile, or costly

The patch is appropriately small. The likely overengineering is a new test that
calls the no-argument validator against today's canonical artifact directory
and expects today's exact failure. Legitimate future evidence would make such a
test brittle and reintroduce the dependency M109 removes.

## 4. What should be kept

- Test-only scope.
- Explicit empty payloads and roots.
- Exactly three positive contract tests.
- Existing deterministic invalid-evidence tests unchanged.
- Full module, full suite, and diff/status inspection.

## 5. Revised recommendation

Precompute the empty evidence, patch the module global to return it, and add an
explicit targeted invalid-evidence test to the validation command. Do not add a
frozen integration expectation against the current canonical artifact history.
