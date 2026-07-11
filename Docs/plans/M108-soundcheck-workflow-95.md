# M108 — Soundcheck Workflow 95

Status: `COMPLETE` — `98/100` (operator-requested local product hardening; does not
replace active external-validation milestone M001)

## Outcome

A production-quality, proposal-only soundcheck workflow with an honest active
frontend and zero new WING write authority.

## Score rubric (100)

| Area | Points | Passing evidence |
| --- | ---: | --- |
| Live safety boundary | 25 | no alternate live writer; zero-write tests |
| Recommendation quality | 25 | non-empty typed absolute proposals; no drift |
| Runtime reliability | 15 | shared services and ownership lifecycle tests |
| Operator UX | 20 | active cockpit, status, blockers, start/stop, errors |
| Verification and truth | 15 | full suites, build, no-WING gate, scoped report |

Target: at least `95/100`. Any unmet criterion remains visible; physical M005
and owner review M006 are scored only in the separate live-release readiness
assessment.

## Execution plan

1. Close CLI/config live-write bypasses.
2. Add explicit measure -> derive -> review pipeline with absolute proposals.
3. Share mixer/audio runtime safely and preserve service ownership.
4. Add the Soundcheck cockpit to the active TypeScript frontend.
5. Prove zero writes, no drift, lifecycle correctness and UI behavior.
6. Run all gates, review the diff, score the result, and continue if below 95.

## Non-goals

- No unattended real-console operation.
- No destructive clean-slate reset.
- No gain/trim boost authority.
- No simulated closure of M005/M006.
