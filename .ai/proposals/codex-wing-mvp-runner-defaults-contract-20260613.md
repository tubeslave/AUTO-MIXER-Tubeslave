# Proposal: WING MVP Runner Defaults Contract

## Short thesis

Bind the WING read-only preflight runner argparse defaults to
`backend/wing_readonly_preflight_contract.py` and have no-WING reject runner
default drift before a real WING session.

## What I understood

The current proof rail depends on a bounded operator recipe: expected plan,
capture command, handoff packet, and the runner itself must point at the same
read-only target. The previous contract work covered generated command text;
the runner parser still had local literal defaults.

## Proposed solution

- Import the shared target default constants in
  `tools/wing_readonly_preflight_runner.py`.
- Use those constants as parser defaults for URL, channels, value types, and
  artifact root.
- Add no-WING introspection of `build_parser().parse_args([])` so runner
  argparse defaults are compared to the same contract.
- Emit JSON/Markdown summary fields for runner argparse target defaults.
- Fail closed on runner argparse target-default mismatch.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Leave runner defaults as local literals. Rejected because the real runner can
  drift from the generated operator commands.
- Only add a runner unit test. Rejected because no-WING should catch the drift
  before an operator follows stale tooling.

## Risks

- Importing the runner from no-WING depends on the same local Python environment
  that already runs the runner tests. If import fails, no-WING fails closed.
- The default channel changes from the runner's old local literal to the shared
  WING MVP proof target. Explicit CLI arguments still override it.

## Test plan

- Compile touched tools.
- Run focused runner/no-WING tests.
- Run no-WING quick with no hardware and no backend writes.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may argue parser defaults are secondary because handoff commands pass
explicit values. The counterpoint is that the runner is the tool used for the
real preflight, so its fallback defaults must stay bound to the same proof
contract.
