# ADR: WING MVP Runner Defaults Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Operator preflight tooling now has shared target defaults for generated plan,
capture, and handoff commands, but the runner parser still owned local literal
defaults.

## Options considered

1. Keep runner argparse defaults local.
2. Bind runner argparse defaults to the shared preflight contract and validate
   those defaults in no-WING.
3. Only add a unit test for runner defaults.

## Decision

Use option 2.

## Why this won

The runner is the tool that performs the real read-only preflight through the
already-running backend. Its default target recipe must be the same contract
used by handoff and no-WING tooling, and no-WING should reject drift before
hardware is involved.

## Rejected alternatives

- Option 1 leaves the physical preflight tool free to diverge.
- Option 3 proves the runner in focused tests but does not expose the mismatch
  in the release/prestart summary.

## Implementation plan

- Import shared target default constants in the runner.
- Use them for `--url`, `--channels`, `--value-types`, and `--artifact-root`
  parser defaults.
- Add a no-WING runner parser-default introspection check.
- Include runner default details in JSON and Markdown summaries.
- Add focused happy-path and stale-default tests.

## Test plan

- Compile touched tools.
- Focused runner and no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter no-WING validation failing in environments where the
runner cannot be imported. That is a safe failure for the prestart gate. Rollback
would restore local runner literals and remove the no-WING parser-default check,
but it would reopen the runner drift path.
