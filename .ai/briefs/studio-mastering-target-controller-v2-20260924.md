# STUDIO Mastering Target Controller v2 — 2026-09-24

## Goal

Turn the accepted mastering meter and safety verdict into a bounded autonomous loudness-target search without giving the machine authority to accept subjective audio.

## Repository finding

`MasteringDirector` already measures integrated LUFS/true peak and rejects limiter-budget regressions, but its gain and maximizer drive are fixed. `loudness_budget_optimizer.py` is a separate older search facade and does not consume the authoritative `MasteringDirector` safety verdict. The next missing link is therefore a small controller that searches real render candidates through the existing safety gate.

## Bounded task

- Add explicit mastering `pregain_db` while measuring protected regressions against the untouched source.
- Search only pregain and maximizer drive; do not tune EQ, transient, stereo, cleanup or mix balance here.
- Run every candidate through `MasteringDirector` and its existing LUFS, dBTP, crest/width/correlation and limiter-GR gate.
- Discard unsafe candidates and return the untouched source when none are feasible.
- Select the lowest-cost machine-safe candidate using loudness error plus limiter-work/drive penalties.
- Keep every selected audible master `pending_human_review` and `baseline_eligible=false`.
- Bound candidate count explicitly so autonomous search cannot fan out without limit.

## Acceptance

- Input arrays remain unchanged.
- A known reachable synthetic loudness target produces at least one machine-safe candidate within tolerance and limiter budgets.
- An unreachable target rolls back exactly to source audio.
- Candidate-count guard fails closed.
- Existing mastering and repository tests remain green.

## Human listening

Synthetic validation can accept the controller mechanics only. No musical master is accepted or promoted by this task; a real rendered master still requires level-matched human A/B.
