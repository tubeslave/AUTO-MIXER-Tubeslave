# Brief: P0 — secret hygiene restoration + fail-loud feedback safety path

**Author:** Kimi (independent architect role, per AGENTS.md)
**Date:** 2026-07-19
**Priority:** P0 (blocks any push to public remote)
**Directive:** operator approved Kimi's P0 remediation scope on 2026-07-19 ("приступай"), parallel work with Codex (GPT) which continues the M110–M122 release-evidence thread.

## Context

Independent full-project review by Kimi on 2026-07-19 (report: `PROJECT_REVIEW.md` in the workspace, score 66/100) found three critical issues:

1. **C1 — tracked `.env` with a real `OPENAI_API_KEY`** (`sk-proj-…` format) on HEAD of the current branch and ~14 other local branches. The 2026-05-16 remediation (commit `0c7752271`: untrack + `.env.example` + `Docs/secrets-rotation.md` + guardrail test `tests/test_repo_secret_hygiene.py`) was **reverted by 5 subsequent merges**; the guardrail test was lost in the same merges, so re-registration went unnoticed. Public remote `github.com/tubeslave/AUTO-MIXER-Tubeslave` does NOT contain the key yet (verified via API on all 25 remote branches), but any `git push --all` leaks it.
2. **C2 — silent failure in the emergency feedback path.** `backend/auto_soundcheck_engine.py`:
   - `_monitor_loop` (~8626): `except Exception as e: logger.debug(...)` — monitor-loop death is invisible at default INFO level.
   - `_handle_feedback_event` notch branch (~8786): bare `except Exception: pass` — a failed emergency notch leaves ringing feedback unsuppressed, unnoticed.
   - `_handle_feedback_event` fader_reduce branch (~8812): same bare `except: pass`.
3. **C3 — no always-on feedback suppression** (`server.py:977` keeps `feedback_detector = None` outside soundcheck) — **out of this brief's scope**, proposed as follow-up M-task for the Council.

## Scope of this brief (Kimi implements)

- `git rm --cached .env` on the current branch (operator rotates the key itself — agents never touch secret values, per AGENTS.md).
- Restore `tests/test_repo_secret_hygiene.py` from `0c7752271`, **strengthened**: original checked only `git check-ignore`; add a `git ls-files` assertion so a tracked `.env` fails CI even when ignore rules exist (the exact regression that recurred).
- Sync `.env.example` with the actually used `AUTOMIXER_LLM_BACKEND` / `AUTOMIXER_LLM_MODEL` / `AUTOMIXER_MODEL_FALLBACKS` (placeholders only).
- Patch `auto_soundcheck_engine.py` feedback path to fail loud: error-level logging with traceback, consecutive-error escalation in the monitor loop, bounded fader fallback (−3 dB, same guards as the existing fader path) when an emergency notch raises.
- ADR in `Docs/adr/`, targeted pytest runs, focused commits. No unrelated dirty files touched (`reports/loop_soundcheck_pipeline.jsonl` and `artifacts/**` belong to the Codex thread).

## Explicit non-goals

- Key rotation (operator action at the OpenAI dashboard).
- Untracking `.env` on the other ~14 branches and history rewrite (`git filter-repo`) — separate task, needs force-push coordination.
- C3 always-on feedback service, repo bloat diet, dependency pinning — P1 backlog from the review.
