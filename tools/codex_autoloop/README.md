# Codex Autoloop LaunchAgent

This folder prepares a local macOS LaunchAgent that can run safe Codex
development cycles for Automixer every 6 hours, with a local model-budget gate
before each cycle.

It is not installed automatically. Review the files first, then install only if
you want the background automation active.

## Files

- `run_codex_cycle.sh` - one-shot wrapper that runs one Codex cycle and exits.
- `codex_cycle_prompt.md` - prompt used by each Codex cycle.
- `model_budget_profile.json` - local daily/weekly/monthly model profile caps.
- `check_model_budget.py` - selects the cheapest safe model profile for one
  cycle and records reservations in `logs/codex_autoloop/model_budget_state.json`.
- `validate_cycle_report.py` - local checker for required cycle-report evidence.
- `../no_wing_release_check.py` - timeout-bounded offline release-check runner
  for backend/frontend gates without WING access.
- `com.tubeslave.automixer.codex-autoloop.plist` - LaunchAgent definition.

## Behavior

- The LaunchAgent attempts one Codex cycle every `21600` seconds.
- Before Codex starts, `check_model_budget.py` selects a model profile from
  `model_budget_profile.json` and reserves one local budget event.
- The budget gate cannot read provider-side tariff usage. It protects the plan
  conservatively by limiting automated requests for expensive profiles:
  `budget_light`, `budget_balanced`, `budget_strict`, and rare `premium_high`.
- If all allowed local profiles are exhausted, the wrapper exits successfully
  without starting Codex so quota is preserved for the next day/week/month.
- Codex limit checks are disabled. The current runner does not use the older
  `CODEX_LIMITS_*` provider-limit files and does not skip runs based on limits;
  the active model-budget profile gate is a separate local cycle-routing guard.
  Current policy supersedes older local reports that described a limit gate.
- The default automated path avoids GPT-5.5 high reasoning. High reasoning is
  reserved for rare architecture/safety review windows and is disabled for
  normal daily automation by the profile.
- The wrapper has a hard project-root guard and refuses to run unless both the
  script path and git root resolve to
  `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`.
- The prompt explicitly forbids work in neighboring projects such as external
  Paperclip/PiperClip folders. Repo-local `.paperclip/reports/` is used only as
  an Automixer report directory.
- The prompt keeps TODO/FIXME scans focused on owned Automixer code, tests,
  tooling, and current docs. Vendored, generated, artifact, and reference
  corpora such as `external/`, `models/training_state.json`,
  `backend/ai/knowledge/`, `frontend/node_modules/`, nested
  `**/node_modules/`, `backend/venv/`, `**/.venv/`, `artifacts/`, and
  generated/log/report-history paths such as `frontend/dist/`,
  `logs/codex_autoloop/`, `.paperclip/reports/`, plus archived/manual docs are
  treated as scan noise unless the selected task explicitly targets them. The
  TODO/FIXME scan should use `rg --hidden --glob ...` directly rather than
  `find ... | xargs rg`, because generated app bundle paths can contain spaces.
- Does not run immediately at login; `launchd` starts it on the 6-hour
  interval.
- Uses a lock directory at `.codex_autoloop.lock` to avoid parallel runs.
- Writes lock owner metadata to `.codex_autoloop.lock/owner.env` after acquiring
  the lock, including pid, parent pid, UTC start time, host, project root, and
  wrapper log path. If another run finds the lock, it prints that metadata when
  available instead of guessing whether the lock is active or stale.
- Writes wrapper logs to `logs/codex_autoloop/`.
- After a successful Codex run, finds the new repo-local cycle report and
  validates the required safety/report shape, including a valid, not-future UTC
  report title timestamp that matches the filename, a short plan section with
  one to three numbered/bulleted items for the selected cycle tasks,
  discovery evidence for the branch/worktree, latest/recent repo-local report
  directories, latest/recent test artifacts, TODO/FIXME scan, local `.ai`
  tasks, and non-empty evidence sections for
  findings, changes, tests, remaining work, risks, and each plain-language
  explanation subsection. Worktree discovery must name the exact
  `git status --short` command and the current branch. Core report sections
  must remain in the expected
  order: scope, findings, short plan, changes, changed files, tests, memory
  lint result, remaining work, and risks. The TODO/FIXME evidence must state
  the owned-code focus and the excluded vendored/generated/reference scan-noise
  boundary. The tests section must state an outcome such as passed, failed,
  skipped, blocked, or not run, not only list commands, and must include a
  memory lint result.
  Reports must also include a dedicated non-empty `## Memory Lint Result`
  section with the `memory_lint.py` command/result and an explicit outcome.
  The `Changed Files` section must be
  non-empty, include repo-local markdown links, and link back to the cycle
  report itself so the audit artifact is visible in the file list. Reports must
  explicitly state the exact Scope lines
  `Codex CLI was the only agent workflow used.` and
  `External Paperclip/PiperClip was not used as an agent runner.`, and must
  explicitly state that no neighboring repositories or external local projects
  were analyzed, modified, or used. Scope must include `Repo-local report path:`
  with `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports` so
  repo-local reports cannot be confused with external Paperclip/PiperClip
  output. The worktree evidence must mention both modified/changed files and
  new/untracked files, even when one side is empty.
  The plain-language project-state subsection must explicitly cover what
  already works, what works partially, what is not implemented, and the current
  risks. The end-of-report order is fixed: `## Три обязательных вопроса`
  appears before the final top-level `## Объяснение простым языком` section,
  and no top-level report section may follow that plain-language section.
  Reports must not contradict the repo-local boundary later by claiming
  that external Paperclip/PiperClip ran the cycle or that neighboring project
  paths such as `/Users/dmitrijvolkov/paperclip` were analyzed, modified, or
  used.
  Forbidden live command tokens, including input-trim/preamp aliases, must be
  listed in an explicitly negative audit sentence, such as a statement that no
  such command was run. Non-destructive
  action terms such as commit, push, reset, destructive cleanup, secret access,
  and user-file deletion must also appear in an explicitly negative audit
  sentence. The validator also rejects reports that include a valid no-action
  audit in `## Scope` and then contradict it in later report sections with an
  explicit executed/run/sent/applied/performed action sentence.
- Direction guard activation is inside each Codex cycle: the prompt requires
  `tools/codex_autoloop/codex_direction_guard.py` before task selection and
  again before the final cycle report. The guard reads git status, recent
  commit subjects, and repo-local reports, then scores whether the cycle is
  moving toward supervised WING Rack MVP blockers or drifting into
  memory/queue/result-consumer/report-only work. The wall-clock cadence remains
  the existing LaunchAgent interval of `21600` seconds when installed; no new
  background service is installed by the guard.
- When WING Rack is connected, the guarded next action starts with
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
  starting the intended backend/WING session, then continues to the read-only
  WING preflight or manual review of a real preflight artifact. This prestart
  gate is diagnostic only and does not stop processes, contact WING, or grant
  live authority.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_gate` exactly matching
  `expected_backend_port_prestart_gate`,
  `backend_port_prestart_gate.grants_live_authority: false`, and
  `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`.
  This first gate proves only that the backend-port prestart check is still
  before the intended backend/WING session; it does not contact WING or grant
  live authority.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate` must also show
  `backend_port_prestart_gate.operator_sequence_step_index: 0` and
  `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`.
  These fields keep the structured gate aligned with the first operator step.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_check_operator_sequence_step_index: 0`,
  `backend_port_prestart_check_must_be_first_operator_step: true`, and
  `backend_port_prestart_check_operator_sequence_step_is_first: true`.
  This proves the backend-port prestart check remains the first operator step
  before any intended backend/WING session.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show
  `supervised_prestart_gate` exactly matching
  `expected_supervised_prestart_gate` and
  `supervised_prestart_gate.grants_live_authority: false`. This structured gate
  is evidence for the proof boundary only and does not grant live authority.
- Before any supervised fader/gain proof, the structured
  `supervised_prestart_gate` must also show
  `supervised_prestart_gate.operator_sequence_step_index: 9` and
  `supervised_prestart_gate.operator_sequence_step_must_be_final: true`.
  These fields keep the supervised gate aligned with the final operator step.
- Before any supervised fader/gain proof, the structured
  `supervised_prestart_gate` must also show
  `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready` and `supervised_prestart_gate.unlocks_runbook:
  Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.
  These fields keep the gate tied to an accepted decision and the supervised
  fader/gain proof runbook only.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show `mvp_blocker_anchor_matches_current_first_open: true`,
  `mvp_blocker_anchor_sha256_matches_current_first_open: true`,
  `mvp_blocker_anchor.packet_does_not_close_blocker: true`,
  `operator_sequence_has_mvp_blocker_anchor_digest_review: true`,
  `operator_sequence_has_supervised_prestart_gate: true`,
  `operator_sequence_supervised_prestart_gate_is_final: true`, and
  `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`.
  These fields prove only that the offline handoff packet still points at the
  current first WING MVP blocker and keeps supervised proof planning behind a
  valid accepted-candidate prestart trace; they do not grant live authority.
- Before any supervised fader/gain proof, the
  `operator_sequence_mvp_blocker_anchor_digest_review_step` must include
  `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
  `current first open blocker`.
- Before any supervised fader/gain proof, the final
  `operator_sequence_supervised_prestart_gate_step` must include
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Project Director Agent activation is inside each Codex cycle after the
  direction guard. `tools/codex_autoloop/project_director_agent.py` reads
  `Docs/MVP_WING_SUPERVISED.md`, `.ai/project-layer/wing_mvp_blockers.md`, and
  the guard result, then selects the first non-closed MVP blocker as the only
  acceptable task direction. It writes `.paperclip/reports/project_director_latest.md`
  and never writes code, contacts WING, or grants live authority.
  If it detects broad-work drift, development still continues in Goal mode; the
  next task is redirected to the first non-closed MVP blocker instead of
  stopping the autonomous cycle.
- `tools/codex_autoloop/update_mvp_progress.py` refreshes
  `Docs/reports/MVP_PROGRESS_LATEST.md` from the blocker table before planning
  and before the final report.
- Focused offline release gates can use `tools/no_wing_release_check.py`; each
  subprocess check has a default timeout and records a `timeout` result instead
  of leaving an autonomous run silent indefinitely.
- Requires `codex` CLI to be available.
- Uses `codex exec -C <project> --sandbox workspace-write --model <selected>`.
- Sets Codex approval policy to `never` for non-interactive execution.
- Sets `model_reasoning_effort` from the selected profile.
- Does not enable live WING/OSC writes. The prompt explicitly requires
  dry-run/audit-only behavior for risky actions.
- Does not commit or push.

## Dry Run

From the project root:

```bash
chmod +x tools/codex_autoloop/run_codex_cycle.sh
CODEX_AUTOLOOP_DRY_RUN=1 tools/codex_autoloop/run_codex_cycle.sh
plutil -lint tools/codex_autoloop/com.tubeslave.automixer.codex-autoloop.plist
```

Dry run still checks the project-root, prompt, validator, and lock contract, but
it does not require a working Codex CLI executable because it never starts
`codex exec`.

The dry run also prints the selected local model profile without reserving a
budget event.

## Model Budget Overrides

The default profile is conservative. For one manual cycle you can override it:

```bash
CODEX_AUTOLOOP_MODEL_PROFILE=budget_light tools/codex_autoloop/run_codex_cycle.sh
CODEX_AUTOLOOP_MODEL_PROFILE=budget_strict tools/codex_autoloop/run_codex_cycle.sh
```

Use `premium_high` only for explicit architecture/safety review. It is capped
by `model_budget_profile.json` and is not part of normal daily automation.

You may override the exact model string if the local Codex installation exposes
a different model name:

```bash
CODEX_AUTOLOOP_MODEL=gpt-5.5 CODEX_AUTOLOOP_REASONING_EFFORT=low \
  tools/codex_autoloop/run_codex_cycle.sh
```

If `shellcheck` is installed:

```bash
shellcheck tools/codex_autoloop/run_codex_cycle.sh
```

## One Manual Real Run

This starts one Codex cycle in the foreground and then exits:

```bash
tools/codex_autoloop/run_codex_cycle.sh
```

## Install LaunchAgent

Only run this when you intentionally want the 6-hour background automation:

```bash
mkdir -p "$HOME/Library/LaunchAgents" logs/codex_autoloop .paperclip/reports
cp tools/codex_autoloop/com.tubeslave.automixer.codex-autoloop.plist \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
launchctl enable "gui/$(id -u)/com.tubeslave.automixer.codex-autoloop"
```

## Inspect Status

```bash
launchctl print "gui/$(id -u)/com.tubeslave.automixer.codex-autoloop"
tail -f logs/codex_autoloop/launchd.out.log
tail -f logs/codex_autoloop/launchd.err.log
```

## Uninstall LaunchAgent

```bash
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
rm -f "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
```

## Notes

- Do not install multiple Codex autoloops for the same repo.
- If Codex.app starts opening unexpectedly, inspect active launchd jobs:

```bash
launchctl list | rg -i "codex|openai|chatgpt|paperclip|telegram|claw"
```

- This LaunchAgent is interval-based. The shell script must not contain its own
  infinite loop.
