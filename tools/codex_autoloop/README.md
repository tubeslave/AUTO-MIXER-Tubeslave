# Codex Autoloop LaunchAgent

This folder prepares a local macOS LaunchAgent that can run safe Codex
development cycles for Automixer every 20 minutes.

It is not installed automatically. Review the files first, then install only if
you want the background automation active.

## Files

- `run_codex_cycle.sh` - one-shot wrapper that runs one Codex cycle and exits.
- `codex_cycle_prompt.md` - prompt used by each Codex cycle.
- `com.tubeslave.automixer.codex-autoloop.plist` - LaunchAgent definition.

## Behavior

- The LaunchAgent starts one Codex cycle every `1200` seconds.
- Codex limit checks are disabled. The wrapper does not inspect quota state and
  does not skip runs based on limits.
- The wrapper has a hard project-root guard and refuses to run unless both the
  script path and git root resolve to
  `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`.
- The prompt explicitly forbids work in neighboring projects such as external
  Paperclip/PiperClip folders. Repo-local `.paperclip/reports/` is used only as
  an Automixer report directory.
- Does not run immediately at login; `launchd` starts it on the 20-minute
  interval.
- Uses a lock directory at `.codex_autoloop.lock` to avoid parallel runs.
- Writes wrapper logs to `logs/codex_autoloop/`.
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
