# ADR: P0 — secret hygiene restoration + fail-loud feedback path

**Date:** 2026-07-19
**Status:** Accepted (operator directive 2026-07-19)
**Authors:** Kimi (implementation), review requested from Codex
**Refs:** `.ai/briefs/p0-secret-hygiene-feedback-failsafe.md`, `.ai/proposals/p0-secret-hygiene-feedback-failsafe-kimi.md`, `Docs/secrets-rotation.md`, commit `0c7752271`

## 1. Context

Full-project review (2026-07-19) found:

- Root `.env` containing a real `OPENAI_API_KEY` was tracked again on HEAD. The 2026-05-16 remediation (`0c7752271`) had removed it, but five merges re-added the file and lost the guardrail test `tests/test_repo_secret_hygiene.py`, so the regression was invisible. Public remote does not contain the key yet; any push of the affected branches would leak it.
- `backend/auto_soundcheck_engine.py` emergency feedback path failed silently: `_monitor_loop` logged exceptions at DEBUG (invisible at default INFO), and both `_handle_feedback_event` branches used bare `except Exception: pass`, so a failed emergency notch or fader reduction left live feedback unsuppressed with no signal to the operator.

## 2. Options considered

A. Full history rewrite (filter-repo/BFG) + force-push now.
B. Untrack `.env` on the current branch, restore and strengthen the guardrail test, rotate the key (operator), defer rewrite.
C. For feedback path: raise and kill the monitor thread vs. log loudly and keep the loop alive; direct fader fallback vs. synthesizing a `fader_reduce` event.

## 3. Decision

- Option B for secrets: `git rm --cached .env`; restore `tests/test_repo_secret_hygiene.py` from `0c7752271` and add `test_env_is_not_tracked_in_git_index` (asserts `git ls-files --error-unmatch .env` fails — catches tracked-but-ignored, the exact regression class); sync `.env.example` with `AUTOMIXER_LLM_*` placeholders. Operator rotates `OPENAI_API_KEY` per `Docs/secrets-rotation.md`.
- For feedback path: fail-loud and stay alive. `_monitor_loop` logs `logger.exception` with a consecutive-error counter and one `logger.critical` escalation at 10 consecutive failures; the loop keeps running. `_handle_feedback_event` notch branch logs the failure and calls the new `_apply_feedback_fader_fallback` (bounded −3 dB, floor −144 dB, same `_control_allowed` gates and `_execute_action` path as before). The fader branch was refactored onto the same helper with loud exception logging. No behavior change on the success path.

## 4. Why this won

- Stops the immediate leak vector (index) without a destructive history rewrite that needs force-push coordination across ~15 branches.
- The strengthened test guards the actual failure mode: ignore rules never untrack files, so the old ignore-only test could not have caught what happened.
- In live sound, a dead monitor loop is worse than a noisy one; visibility + bounded fallback matches the project canon ("при сомнениях — снижай", feedback first).

## 5. Rejected alternatives

- Immediate filter-repo: correct but requires coordinated force-push; tracked as follow-up.
- Killing the monitor loop on error: removes protection entirely.
- Synthesizing a `FeedbackEvent` for the fallback: obscures causality in logs.

## 6. Implementation plan

Done in this change: untracked `.env` (index), restored+strengthened test, `.env.example` sync, three patches in `auto_soundcheck_engine.py` (+1 `__init__` counter). Follow-ups (not in this ADR): key rotation by operator; untrack `.env` on the other branches; gitleaks in CI + pre-push hook; always-on feedback service in `server.py` (review finding C3); history rewrite.

## 7. Test plan

Executed: `py_compile` on the engine; `pytest tests/test_repo_secret_hygiene.py tests/test_auto_soundcheck_engine.py tests/test_auto_soundcheck_engine_safety.py` — **100 passed** in `backend/venv` (Python 3.14.3). The strengthened test passes only while `.env` is untracked, which is the intended guard.

## 8. Risks and rollback

- Fallback double-apply: fallback runs only inside the notch exception handler and shares caps/cooldowns via `_execute_action` and `_record_feedback_action_sent`.
- New critical log could spam if the loop is permanently broken: one-time escalation at exactly 10 consecutive errors; per-iteration entries remain at ERROR.
- Rollback: revert the two commits (security, safety). The `.env` file itself was never deleted from disk; re-tracking is `git add .env` (explicitly not recommended).
