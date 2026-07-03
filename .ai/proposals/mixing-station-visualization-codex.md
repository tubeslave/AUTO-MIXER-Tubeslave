# Mixing Station Visualization — Codex Proposal

## Short thesis

Implement Mixing Station as a separate, disabled-by-default visualization backend
that mirrors already safety-checked Automixer decisions and refuses unknown or
dangerous live writes.

## What I understood

The goal is not aggressive real-console control. The immediate goal is offline
or dry-run visualization in Mixing Station Desktop on a MacBook, with future
REST/WebSocket/OSC control only after dataPath discovery and explicit safety flags.

## Proposed solution

Add `backend/integrations/mixing_station/` with:

- `AutomixCorrection` as the shared correction format.
- capability maps for `wing_rack` and `dlive`.
- mapping files from Automixer parameters to Mixing Station dataPaths where known.
- REST/WebSocket transport clients with clear unavailable/discovery errors.
- OSC fallback using `/con/v/{dataPath}` and `/con/n/{dataPath}`.
- a safety layer with dry-run defaults, clamping, rate limits, emergency stop,
  and blocked destructive actions.
- JSONL logging at `logs/automix_to_mixing_station.jsonl`.

Wire `AutoSoundcheckEngine._execute_action()` to optionally mirror
`SafetyDecision` objects into the adapter when `config/mixing_station.yaml`
has `enabled: true`.

## Likely files to touch

- `backend/auto_soundcheck_engine.py`
- `backend/integrations/mixing_station/*`
- `config/mixing_station.yaml`
- `config/mixing_station/*`
- `scripts/*mixing_station*.py`
- `tests/test_*mixing_station*` or equivalent requested test files
- `docs/MIXING_STATION_INTEGRATION.md`
- `Docs/adr/mixing-station-visualization.md`

## Alternatives considered

- Reusing `WingClient`/`DLiveClient` directly: rejected because it couples
  visualization to live-console control and makes dry-run safety less explicit.
- Adding commands directly to `server.py`: rejected by project architecture.
- Assuming full Mixing Station dataPaths: rejected because the local API Explorer
  must be the authority.

## Risks

- Mixing Station REST/WebSocket write endpoints are not known until local API
  discovery. The adapter should block live writes when endpoint/path information
  is missing.
- Some existing legacy agents still write directly through mixer clients. The
  first safe integration should target the modern AutoFOH safety path and document
  the legacy paths.

## Test plan

- Unit-test mapping and capabilities.
- Unit-test safety clamps, emergency stop, forbidden scene/phantom actions.
- Unit-test dry-run adapter behavior and logging.
- Unit-test REST unavailable message and OSC address generation.
- Run targeted tests first, then the standard suite if feasible.

## Where the other agent may disagree

Kimi may argue for a broader console abstraction that also wraps old direct
`mixer.send()` agents. That is architecturally desirable, but higher risk for
this first phase. A mirror after `AutoFOHSafetyController` is the safer initial
move.
