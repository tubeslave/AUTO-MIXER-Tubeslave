# Implementation Spec: Dante Routing Scheme UI

## Goal
Show users exactly what signals the program expects on each Dante channel.

## Backend (DONE)
- `/backend/dante_routing_config.py` — routing scheme config
- `server.py` — `get_dante_routing` endpoint added
- `websocket.js` — `getDanteRouting()` method added

## Frontend Changes Needed

### 1. App.js — Connect Page Redesign
After the audio device selector, add a **Routing Scheme** card that shows the expected Dante layout.
Load routing info on mount via `websocketService.getDanteRouting()`.

The routing scheme card should show:
- A visual table/grid of channel ranges with color-coded roles
- Each range shows: channel numbers, signal type, tap point, required/optional badge
- WING routing hints for each range
- Expandable details for modules that use each range

Color scheme:
- CHANNEL_ANALYSIS (blue #00d4ff) — main channels
- CHANNEL_DRY (orange #f0883e) — dry soundcheck channels
- MASTER (red #f85149) — master bus
- DRUM_BUS (green #3fb950) — drum group
- VOCAL_BUS (purple #a371f7) — vocal group
- INSTRUMENT_BUS (yellow #d29922) — instrument group
- MEASUREMENT_MIC (cyan #39d353) — measurement
- AMBIENT_MIC (pink #f778ba) — ambient
- MATRIX (gray #8b949e) — matrix
- RESERVE (dark gray #484f58) — reserve

### 2. Each Module Tab — Signal Hint Banner
At the top of each module (before the action buttons), add a small info banner showing what signal the module expects. Use `MODULE_SIGNAL_INFO` from the routing config.

Format: A compact, colored bar with icon + signal type + brief description.
Style: `.signal-hint` — small banner, subtle background, no border, icon on left.

### 3. CSS additions needed in App.css
- `.routing-scheme` — container for the routing table
- `.routing-row` — each channel range row
- `.routing-badge` — required/optional badge
- `.routing-role` — color-coded role indicator
- `.signal-hint` — module signal banner
- `.wing-hint` — WING routing instruction text
