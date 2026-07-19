# Proposal: P0 — secret hygiene restoration + fail-loud feedback path (Kimi)

## 1. Short thesis

Re-apply the lost secret-hygiene guardrail in a form that survives the regression class that defeated it (ignore-rules present but file tracked anyway), and convert the emergency feedback path from silent-failure to fail-loud with the project's canonical fallback ("при сомнениях — снижай").

## 2. What I understood

- The 2026-05-16 fix (`0c7752271`) worked, but 5 merges brought `.env` back into the index and dropped `tests/test_repo_secret_hygiene.py`; nothing was left to detect it.
- `_monitor_loop` swallowing exceptions at DEBUG means the only always-on feedback watcher (during soundcheck) can die silently.
- In `_handle_feedback_event`, both `except Exception: pass` blocks sit on the emergency path where the cost of an unnoticed failure is hearing damage.
- `_apply_direct_feedback_emergency_notch` already fails loud (warning + `False`) — it is the correct in-file pattern to match.

## 3. Proposed solution

1. `git rm --cached .env`; commit the index deletion (the local file stays; operator rotates the key).
2. Restore the guardrail test from `0c7752271` and add `test_env_is_not_tracked_in_git_index` asserting `git ls-files --error-unmatch .env` fails. Keep the original ignore-rule assertions unchanged.
3. Append the three `AUTOMIXER_LLM_*` keys to `.env.example` with placeholder values.
4. `_monitor_loop`: `logger.exception` (ERROR + traceback), count consecutive failures, reset on success, `logger.critical` once at 10 consecutive failures ("feedback monitoring degraded — operator attention required"). Loop stays alive.
5. `_handle_feedback_event` notch branch: on exception, `logger.exception` + attempt bounded fader fallback (−3 dB floor −144, gated by the same `_control_allowed(info, "emergency_fader"/"fader")` check) so a failed notch does not leave feedback unsuppressed. Fader branch: same loud logging; no behavior change beyond visibility.

## 4. Likely files to touch

- `.env` (index only), `.env.example`
- `tests/test_repo_secret_hygiene.py` (restored + strengthened)
- `backend/auto_soundcheck_engine.py` (3 small patches + 1 attr init)
- `Docs/adr/p0-secret-hygiene-feedback-failsafe.md`, `.ai/` records

## 5. Alternatives considered

- **BFG/filter-repo now** — rejected for this brief: history rewrite needs force-push coordination with the operator and all branches; untracking stops the bleeding first.
- **Raise instead of log in monitor loop** — rejected: killing the monitor thread is worse than a noisy loop; fail-loud + stay alive.
- **Refactor feedback handling into a strategy class** — rejected: violates smallest-safe-diff before the M-release thread finishes.

## 6. Risks

- Fader fallback on a failed notch could double-apply if both paths race — mitigated: fallback runs only inside the notch exception handler, uses the same caps/cooldowns via `_execute_action`, and `_record_feedback_action_sent` bookkeeping is shared.
- Strengthened test runs `git` in CI — the existing restored test already shells out to git; runner images have git (CI checkout action). No new dependency.
- `.env.example` placeholders must not look like real keys — using `your-...-here` style.

## 7. Test plan

- `backend/venv/bin/python -m pytest tests/test_repo_secret_hygiene.py -v`
- `backend/venv/bin/python -m pytest tests/test_auto_soundcheck_engine.py tests/test_auto_soundcheck_engine_safety.py -q`
- Spot-check: `git ls-files .env` empty after `git rm --cached`.

## 8. Where the other agent may disagree

- Codex may prefer routing the fallback through the existing `fader_reduce` event path instead of a direct call; I chose a direct guarded call because synthesizing a `FeedbackEvent` inside the exception handler obscures causality in logs.
- Codex may want the strengthened test to scan all tracked files for secret patterns (gitleaks-style); I kept it to the `.env` index check as the minimal regression guard and listed gitleaks-in-CI as P1 in the review.
