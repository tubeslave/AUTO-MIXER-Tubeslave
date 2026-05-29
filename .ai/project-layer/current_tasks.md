# Current Tasks

## Purpose

This file is a lightweight task memory for agents. It does not replace
Paperclip, ADRs, tests, or git history.

## Active Tasks

- Maintain the memory pipeline and keep safety policy explicit.
- Continue improving `input_trim_preamp_character` as the canonical input-trim
  method when the task is specifically about gain staging.
- Keep Paperclip/Codex automation in dry-run or audit-only mode unless the
  operator explicitly approves a supervised action.
- Preserve Council artifacts in `.ai/briefs/`, `.ai/proposals/`, `.ai/reviews/`,
  and `Docs/adr/`.

## Watch List

- Do not let report automation become a WING/OSC write path.
- Do not treat shadow-only or offline-only evidence as live readiness.
- Keep memory append-only unless a human explicitly asks for cleanup.
- If full tests fail for old reasons, run the focused test and record the wider
  failure honestly.

## Update Rule

Append task updates with date, owner, status, evidence, and next action. Do not
delete old tasks unless a human explicitly requests pruning.
