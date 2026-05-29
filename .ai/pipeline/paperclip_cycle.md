# Paperclip Cycle

## Purpose

Paperclip should create and consume Automixer work without bypassing safety or
turning issue text into console authority.

## Cycle

READ PROJECT LAYER
-> ACCEPT TASK
-> CLASSIFY TASK
-> CHECK SAFETY
-> EXECUTE ONLY ALLOWED ACTIONS
-> WRITE STRUCTURED REPORT
-> UPDATE CURRENT TASK STATUS
-> NEVER PERFORM LIVE OSC WRITE WITHOUT EXPLICIT APPROVAL

## Task Classification

Every task should declare one mode:

- `audit-only`
- `dry-run`
- `supervised-write`
- `live-write`

When unclear, choose audit-only or dry-run.

## Allowed Actions

Paperclip automation may:

- read project memory;
- create or update issues;
- run safe analysis;
- request Codex work;
- write structured reports;
- update task status.

Paperclip automation must not:

- send live OSC writes;
- mutate WING, Dante, or live audio state;
- bypass operator approval;
- promote a recommendation into a control command.

## Report Requirements

Reports must include:

- task mode;
- files inspected or changed;
- tests or checks run;
- safety status;
- whether live OSC/WING writes occurred;
- next steps.
