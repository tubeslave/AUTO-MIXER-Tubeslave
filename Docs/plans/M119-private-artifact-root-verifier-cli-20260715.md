# M119: Private Artifact Root Verifier CLI

## Boundary

Offline migration-tool and truth-control changes only. No backend transport,
CoreAudio, OSC or WING contact; no live authority widening.

| Checkpoint | Goal | Status |
|---|---|---|
| M119-CP01 | Reproduce CLI/API discrepancy | complete |
| M119-CP02 | Council and confinement review | complete with Kimi 402 fallback |
| M119-CP03 | Minimal CLI forwarding fix | complete |
| M119-CP04 | Focused and migration validation | complete |
| M119-CP05 | Full regression and independent review | complete |
| M119-CP06 | Source commit and dynamic CP05 rebind | pending |
| M119-CP07 | Fresh challenge and private-root sealed replay | pending |

## Acceptance

- `--verify-summary --artifact-root <root>` uses that exact root.
- A direct-child run below a real root passes the confinement check.
- Wrong or broad ancestor roots fail closed.
- Symlinked root or run-directory paths fail closed.
- No verifier, durable-store or live-safety policy is weakened.
- The final source-bound replay has exact-zero I/O and writer counters.
