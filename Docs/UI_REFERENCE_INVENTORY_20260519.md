# Automixer UI Reference Inventory - 2026-05-19

Source images:
- `Docs/ui-references/automixer-dashboard-overview.jpg`
- `Docs/ui-references/automixer-agents-console.jpg`
- `Docs/ui-references/automixer-channel-list.jpg`
- `Docs/ui-references/automixer-system-connection.jpg`
- `Docs/ui-references/automixer-reference-board.jpg`

## 1. Tabs / Views

Primary routes required for the prototype:
1. Dashboard / Overview
2. Agents Console
3. Communications
4. Task Queue
5. Memory / Knowledge Base
6. Performance
7. Settings
8. Channel List
9. Groups / Buses
10. System Connection
11. Audio Routing
12. OSC / MIDI
13. Devices
14. Logs
15. Training Sessions
16. Rules Library

## 2. Panels Per View

- Dashboard / Overview: master bus, spectrum, loudness, mix balance, AI confidence, channel strip bank, phase scope, correlation, spectral balance, dynamics overview, loudness history, suggestions timeline, AI director side panel.
- Agents Console: summary metric cards, agent table, agent communications network, task queue, agent messages.
- Communications: agent inbox/sidebar, communications log table, active conversations, director broadcast.
- Task Queue: priority filters, queue table, task detail/action panel.
- Memory / Knowledge Base: memory metrics, memory type distribution, timeline chart, top memories, recent updates.
- Performance: agent performance table, decisions-per-minute chart, success-rate chart.
- Settings: system defaults, automation settings, mix settings, toggles and save/reset actions.
- Channel List: channel summary cards, channel table, selected-channel inspector, level meter.
- Groups / Buses: bus cards/table, group routing, bus detail panel.
- System Connection: topology graph, connection status, connected devices, network interfaces, latency, sync, connection log.
- Audio Routing: inputs, channel/bus mapping, outputs, patch-cable style routing lines.
- OSC / MIDI: device list, live OSC/MIDI messages, settings/status cards.
- Devices: audio/controller/network devices table, selected WING rack panel.
- Logs: log table, filters, log detail tail.
- Training Sessions: session cards/table, dataset status, training queue.
- Rules Library: rule categories, rules table, selected rule inspector.

## 3. Reusable Components

- AppShell: fixed 16:9 desktop console shell with sidebar, top status bar, main work area, optional right panel, bottom transport/action bar.
- Sidebar: grouped navigation with compact icons, active state, system health block.
- TopStatusBar: session, status, mode, safe, connected target, profile/actions.
- BottomTransportBar: play, elapsed, markers, record/snapshot, automation, safe limit, step, rollback.
- MetricCard / Panel: dark technical cards with small uppercase labels.
- DataTable: dense table rows, sticky-like header styling, compact action cells.
- ChannelStrip: vertical console strip with mini waveform, gate/comp labels, meter, fader, mute/solo buttons.
- AgentCard / Agent row: icon, role, status, task, confidence donut, sparkline.
- LogRow / StatusBadge: compact semantic state indicators.
- Sparkline / LevelMeter / Knob / NetworkGraph: SVG/CSS placeholders for realtime-looking visualizations.

## 4. Layout Grid

- Target aspect: 16:9 desktop, optimized around 1280x720 and 1440x810.
- Global frame: left sidebar ~128px, top status bar ~48px, bottom transport bar ~58px.
- Main body: central work area plus right inspector panel ~250px where present.
- Gutter: 8px between panels; panel radius 6-8px; dense internal padding 8-12px.
- Dashboard: top metric row, middle channel bank, bottom analytic row, right AI director stack.
- Table views: summary cards on top, main table center-left, right detail/summary column.
- System/routing views: large visual graph/canvas center, right status/list panels, bottom utility cards.

## 5. Visual Tokens

- Background: `#070b10`, `#0b1118`.
- Panels/cards: `#111923`, `#151f2b`, inner rows `#0d141d`.
- Borders: `#1d2a36`, brighter active border `#2b3c4f`.
- Text: primary `#e6edf3`, muted `#8b949e`, dim `#5f6b78`.
- Green online/applied: `#2dd36f`, `#3fb950`.
- Orange warning/pending: `#f0a22e`, `#d29922`.
- Red danger/rejected: `#f85149`.
- Blue audio/OSC: `#2f9bff`, `#00d4ff`.
- Purple AI/agent: `#a371f7`, `#c084fc`.
- Typography: compact sans-serif, uppercase section labels, 10-12px labels/table cells, 18-28px metric numerals.

## 6. Interactive States

- Sidebar item: default muted, hover blue-tinted, active bright cyan left rail/background.
- Top mode/status chips: dropdown-like pills with semantic color dot.
- Tables: selected row darker/brighter border, hover row highlight, action icons visible.
- Buttons: primary green play/save, dark ghost buttons, red destructive, disabled dim.
- View tabs: active underline/cyan bar, inactive muted labels.
- Channel strip: selected channel purple outline; mute/solo buttons have red/yellow semantic states.
- Proposal/task states: high red, medium orange, low green; applied green, pending orange, rejected red.
- Prototype constraint: all interactions are local mock state only; no OSC, no websocket, no live writes.
