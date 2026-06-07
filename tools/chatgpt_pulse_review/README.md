# ChatGPT Pulse Review Automation

This folder defines a macOS LaunchAgent that runs one safe Automixer Pulse
review every day at 09:00 local time.

The job:

1. Fetches the current ChatGPT Pulse through the local logged-in Chrome session.
2. Saves a sanitized JSON/Markdown snapshot under `logs/chatgpt_pulse_review/`.
3. Runs one Codex cycle with `pulse_review_prompt.md`.
4. Asks Codex to accept only recommendations that are useful for Automixer,
   put accepted items into the project plan/briefs, and implement at most one
   small safe slice.

Safety boundary:

- No live OSC/WING writes.
- No real mixer connection.
- No `git commit`, `git push`, or destructive cleanup.
- No `.env`, tokens, credentials, or API keys are logged.
- The fetcher writes only sanitized Pulse card text and short conversation
  excerpts; it does not write cookies, access tokens, session payloads, account
  identity fields, or signed image URLs.
- The wrapper uses the same `.codex_autoloop.lock` as the Codex autoloop,
  so two Codex automations do not write the repo at the same time.

## Files

- `fetch_chatgpt_pulse.py` - fetches a sanitized Pulse snapshot.
- `pulse_review_prompt.md` - prompt used by the daily Codex review.
- `run_pulse_review_cycle.sh` - one-shot wrapper; launchd calls this.
- `com.tubeslave.automixer.chatgpt-pulse-review.plist` - LaunchAgent.

## Dry Run

```bash
chmod +x tools/chatgpt_pulse_review/run_pulse_review_cycle.sh
PULSE_REVIEW_DRY_RUN=1 tools/chatgpt_pulse_review/run_pulse_review_cycle.sh
plutil -lint tools/chatgpt_pulse_review/com.tubeslave.automixer.chatgpt-pulse-review.plist
python3 -m py_compile tools/chatgpt_pulse_review/fetch_chatgpt_pulse.py
```

If `shellcheck` is installed:

```bash
shellcheck tools/chatgpt_pulse_review/run_pulse_review_cycle.sh
```

## One Manual Run

This fetches Pulse and runs one Codex review cycle immediately:

```bash
tools/chatgpt_pulse_review/run_pulse_review_cycle.sh
```

## Install LaunchAgent

The LaunchAgent does not require sudo.

```bash
mkdir -p "$HOME/Library/LaunchAgents" logs/chatgpt_pulse_review .paperclip/reports
cp tools/chatgpt_pulse_review/com.tubeslave.automixer.chatgpt-pulse-review.plist \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.chatgpt-pulse-review.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.chatgpt-pulse-review.plist"
launchctl enable "gui/$(id -u)/com.tubeslave.automixer.chatgpt-pulse-review"
```

## Inspect Status

```bash
launchctl print "gui/$(id -u)/com.tubeslave.automixer.chatgpt-pulse-review"
tail -f logs/chatgpt_pulse_review/launchd.out.log
tail -f logs/chatgpt_pulse_review/launchd.err.log
```

## Uninstall LaunchAgent

```bash
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.chatgpt-pulse-review.plist"
rm -f "$HOME/Library/LaunchAgents/com.tubeslave.automixer.chatgpt-pulse-review.plist"
```

## Notes

- The daily trigger uses `StartCalendarInterval` at 09:00, not the Codex
  autoloop interval.
- If the local ChatGPT session expires, the fetcher will fail safely and log the
  reason. Open ChatGPT in Chrome and sign in again.
- If another Codex automation is already running at 09:00, this job exits
  without starting a second writer.
