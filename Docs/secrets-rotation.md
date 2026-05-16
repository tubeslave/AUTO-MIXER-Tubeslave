# Secrets Rotation Runbook

This repository previously tracked a root `.env` file in git history. Treat any real secret stored there as exposed until rotated.

## Scope

- Root `.env` now remains local-only.
- Root `.env.example` is the only committed env template at repository root.
- Existing tool-local examples such as `tools/paperclip_watchdog/.env.example` remain valid.

## Immediate remediation

1. Rotate every real credential that was ever stored in the tracked root `.env`.
2. Replace the local `.env` contents with the rotated values.
3. Confirm the new `.env` stays untracked with `git status --short`.
4. Audit logs, screenshots, and shell history for pasted tokens and remove them if present.

## Current root env keys

- `OPENAI_API_KEY`
- `AUTOMIXER_LLM_BACKEND`
- `AUTOMIXER_LLM_MODEL`
- `AUTOMIXER_MODEL_FALLBACKS`

Only `OPENAI_API_KEY` is a credential. The remaining variables are configuration, but they should still stay local to avoid accidental churn.

## Rotation procedure for `OPENAI_API_KEY`

1. Create a new API key in the provider console.
2. Update the local root `.env` from `.env.example` or edit the existing file in place.
3. Restart any local process that reads the key at startup.
4. Run a dry check before any live console workflow:
   - backend services start successfully
   - no token appears in logs or terminal output
   - no OSC write path is triggered as part of the credential check
5. Revoke the old API key after the new key is verified.

## Verification commands

```bash
git status --short
git check-ignore -v .env .env.local
git check-ignore .env.example && echo "unexpected: example ignored"
```

Expected result:

- `.env` is present locally but not tracked.
- `.env` and `.env.local` are ignored.
- `.env.example` is not ignored.

## Optional follow-up

If this repository was pushed to a shared remote while the real key was present, consider a separate history-rewrite task after rotation. Rotation is the urgent fix; history cleanup can follow under a controlled plan.
