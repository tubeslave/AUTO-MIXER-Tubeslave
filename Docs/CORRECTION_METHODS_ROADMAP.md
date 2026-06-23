# Correction Methods Roadmap

Status: target method-development roadmap

## Purpose

Automixer must develop real correction methods for the mixer console, not only
proof tooling. This roadmap preserves the full product direction while keeping
live writes gated by the supervised WING MVP.

## Current Rule

Before the supervised WING Rack MVP is proven, correction methods may be built
only as:

- offline analysis;
- dry-run recommendation;
- proposal-only payload;
- read-only evidence;
- tests and operator-visible explanations.

Live apply remains limited to the existing supervised fader/gain path and only
after the required WING proof gate.

Correction-method work must not widen live writes before a separate ADR,
hardware proof, rollback proof, emergency-stop proof, and operator approval.

## Method Families

| Family | Examples | Current live status | Development target before MVP proof |
| --- | --- | --- | --- |
| Gain / Trim | input trim, preamp gain, channel gain-down, overload cut | supervised fader/gain only; gain still needs current proof | canonical bounded recommendation, physical readback, rollback evidence |
| Fader / Balance | channel fader, group balance, DCA/main balance | supervised fader only | Ayaic level-plane recommendation, proposal payloads, rollback-safe deltas |
| Pan / Stereo Placement | pan, stereo width, mono compatibility | no live writes | offline/proposal-only pan suggestions with phase/mono safety |
| EQ | corrective EQ, feedback notch, masking cuts, high-pass | no live writes | proposal-only EQ with band identity/readback and feedback/manual-band protection |
| Compressor | ratio, threshold, attack/release, makeup gain | no live writes | proposal-only dynamics with GR, true-peak, and makeup safety |
| Gate / Expander | bleed control, noise floor, hysteresis | no live writes | proposal-only gate settings with source-identity and chatter safety |
| FX / Sends | vocal reverb, delay, monitor-safe sends | no live writes | proposal-only FX send plans and return-level safety |
| Routing / Groups | subgroup assignment, matrices, buses | no live writes | read-only maps and proposal-only routing plans |
| Scenes / Presets | snapshots, scenes, user presets | no live writes | read-only inventory and manual import/export guidance |
| Master / Output | master bus, limiter, matrices, PA feed | no live writes | offline/proposal-only level-plane, true-peak, and PA-feed-safe recommendations |

## Promotion Ladder

Every correction method must move through these stages:

1. `offline_method`: algorithm works on offline or fixture data.
2. `dry_run_recommendation`: method emits bounded recommendation and blockers.
3. `proposal_only`: method emits operator-readable proposal without live apply.
4. `readback_contract`: method knows what physical console state must be read.
5. `supervised_candidate`: method has proof plan, rollback plan, and emergency-stop boundary.
6. `supervised_live_allowed`: only after ADR, tests, real WING proof, and operator approval.

No method may skip stages.

## Acceptance Checklist

A correction method is useful only when it has:

- source evidence and confidence;
- bounded delta;
- safety blockers;
- true-peak/headroom/feedback checks when level can increase;
- physical readback contract;
- rollback expectation;
- operator-readable explanation;
- tests;
- clear live status.

## Current Priority

The first live-capable method family remains fader/gain under the supervised
WING MVP. Other families should continue as offline/proposal/read-only work only
when they do not block the first open WING MVP blocker.
