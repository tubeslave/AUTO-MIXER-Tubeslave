# Codex Autoloop LaunchAgent

This folder prepares a local macOS LaunchAgent that can run safe Codex
development cycles for Automixer every 20 minutes.

It is not installed automatically. Review the files first, then install only if
you want the background automation active.

## Files

- `run_codex_cycle.sh` - one-shot wrapper that runs one Codex cycle and exits.
- `codex_cycle_prompt.md` - prompt used by each Codex cycle.
- `validate_cycle_report.py` - local checker for required cycle-report evidence.
- `../no_wing_release_check.py` - timeout-bounded offline release-check runner
  for backend/frontend gates without WING access.
- `com.tubeslave.automixer.codex-autoloop.plist` - LaunchAgent definition.

## Behavior

- The LaunchAgent starts one Codex cycle every `1200` seconds.
- Codex limit checks are disabled. The wrapper does not inspect quota state and
  does not skip runs based on limits.
- Current policy supersedes older local reports that described a limit gate:
  this checkout has no `check_codex_limits.py`, no `codex_limits.example.json`,
  and no limit-state environment contract in the runner or plist. Reintroducing
  limit gating should be a separate reviewed change with matching tests and
  docs.
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
  archived/manual docs are treated as scan noise unless the selected task
  explicitly targets them.
- Does not run immediately at login; `launchd` starts it on the 20-minute
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
- Focused offline release gates can use `tools/no_wing_release_check.py`; each
  subprocess check has a default timeout and records a `timeout` result instead
  of leaving an autonomous run silent indefinitely.
- Requires `codex` CLI to be available.
- Uses `codex exec -C <project> --sandbox workspace-write`.
- Sets Codex approval policy to `never` for non-interactive execution.
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

Only run this when you intentionally want the 20-minute background automation:

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
