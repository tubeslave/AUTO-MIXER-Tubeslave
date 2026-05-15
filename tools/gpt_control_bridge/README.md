# GPT Control Bridge

Local read-only status bridge for Automixer/Paperclip/watchdog context.

The bridge exposes only GET endpoints and binds to loopback by default. It does not expose dispatch, wakeup, WING, OSC, GitHub, or filesystem write actions.

## Endpoints

- `GET /health`
- `GET /v1/context/automixer`

`/v1/context/automixer` returns Paperclip health, dashboard/agent/run summaries when available, `.paperclip/watchdog_state.json` summary, latest watchdog report path/excerpt, a recommended next action, and safety flags.

## Environment

```bash
PAPERCLIP_API_URL=http://127.0.0.1:3100
PAPERCLIP_COMPANY_ID=<optional-company-id>
AUTOMIXER_PROJECT_ROOT=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS=3
```

If `PAPERCLIP_COMPANY_ID` is omitted, the bridge tries to resolve the first company through `GET /api/companies`. If Paperclip is unavailable, the context endpoint still returns watchdog file summaries and safe error fields.

## Run

```bash
cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
python3 tools/gpt_control_bridge/server.py --host 127.0.0.1 --port 8788
```

## Check

```bash
curl -s http://127.0.0.1:8788/health
curl -s http://127.0.0.1:8788/v1/context/automixer
```

## Safety

- `read_only=true`
- `write_routes_enabled=false`
- `dispatch_allowed=false`
- `wing_osc_allowed=false`
- write methods return `405`
- non-loopback bind hosts are rejected
- no secrets are required or stored by this bridge
