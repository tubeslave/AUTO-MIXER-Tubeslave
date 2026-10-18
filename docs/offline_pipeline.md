# Offline pipeline

This document defines what belongs to offline/research processing and how it may interact with live code safely.

## Purpose

The offline pipeline is for:

- rendering candidate mixes;
- comparing A/B states;
- evaluating full mix, stems, or individual channels;
- testing MuQ-Eval or fallback critics;
- generating reports and training rewards;
- exploring new strategies before live promotion.

## Offline-only rule

Offline modules may render, score, rank, and explain. They must not send OSC to a physical mixer.

```text
multitrack/stems/reference
  -> offline renderer
  -> candidate actions
  -> candidate render
  -> critic/reward model
  -> selected best state
  -> report/logs
```

No live mixer writes are allowed in this path.

## Allowed offline outputs

- WAV/MP3 candidate renders
- `best_mix.wav`
- JSON reports
- JSONL step logs
- score curves
- action timelines
- proposed typed actions for later review

## Forbidden offline outputs

- direct OSC commands to Wing/dLive
- direct mutation of live mixer state
- auto-apply to live soundcheck without safety gate
- frontend controls that bypass backend safety

## Promotion from offline to live

An offline idea may become live-capable only after:

1. It outputs typed actions instead of raw OSC.
2. It has deterministic safety limits.
3. It passes regression tests on known multitracks.
4. It is routed through AutoFOH safety controller.
5. It can run in observation mode.
6. It has rollback behavior for risky changes.
7. It has logs explaining each decision.

## MuQ-Eval role

MuQ-Eval is a critic/reward layer, not a live actuator. It may:

- score before/after candidates;
- reject unsafe or low-quality candidates in offline tests;
- provide reward logs;
- help compare mixes.

It must not:

- own the physical mixer client;
- send OSC;
- become the only live decision-maker;
- override safety guards.

## Candidate rendering

Candidate rendering may explore risky ideas because it works on copied audio files. These risky ideas must not leak into live code unless explicitly converted into bounded typed actions.

## Expected artifacts

Recommended offline artifact layout:

```text
sessions/offline/<session_id>/
  renders/
    candidate_0001.wav
    candidate_0002.wav
    best_mix.wav
  logs/
    steps.jsonl
    decisions.jsonl
    rewards.jsonl
  reports/
    report.json
    report.md
  plots/
    score_curve.png
    action_timeline.png
```

## Boundary tests

The repository should include tests that fail if offline modules import or instantiate direct OSC clients.

Search targets:

```text
SimpleUDPClient
send_message
send_osc
pythonosc
udp_client
WingClient
EnhancedOSCClient
```

Offline modules should be allowed to emit proposed actions, not send them.
