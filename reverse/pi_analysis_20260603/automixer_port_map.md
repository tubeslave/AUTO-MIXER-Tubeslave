# Automixer Port Map From Pi Observations

This is a design map, not a clone of Sound Radix Pi. It translates observable
Pi architecture into safe Automixer-owned behavior.

## Main Takeaway

Pi is relevant to EQ because phase problems often sound like EQ problems.
If two related microphones cancel each other, the mix may lose low end or sound
hollow. A normal EQ boost can make that worse. A phase-aware automixer should
detect the phase/group problem first, then decide whether EQ is still needed.

## Observed Pi Surface To Automixer Equivalent

| Observed Pi surface | Automixer-owned equivalent |
| --- | --- |
| `Group` | source/stem group: drums, bass, guitars, vocals, playback |
| `Group Enable Mode` | group phase-analysis mode: off, internal-only, full, phase-lock |
| `GROUP_INTERNAL_ONLY` | align only channels inside one stem, for example snare top/bottom or drum close mics |
| `GROUP_FULL` | compare the group against the wider mix context |
| `GROUP_PHASE_LOCK` / `Phase-Lock` | preserve a chosen anchor and only propose moves for non-anchor channels |
| `Group Gain` | group-level energy context, not an automatic fader write |
| `Gain` | per-source analysis level, not automatic artistic balance |
| `Weight` | priority/confidence: anchor channels get higher weight |
| `Global Low Freq Focus` / `is_low_freq_focus` | special low-end phase guard for kick, bass, toms, low keys, and playback |
| `lookahead_buffer_ms` | offline/soundcheck lookahead window; live needs latency reporting and strict bounds |
| `report_latency` / `logic_latency` | latency budget metadata surfaced to operator/product state |
| `is_group_graph` / `is_freq_graph` | reporting views: group coherence and frequency-zone phase risk |

## Proposed Automixer Method Boundary

Method id:

```text
automixer_phase_aware_eq_guard_v1
```

Modes:

- offline render: can apply delay/polarity candidates before EQ if peak/loudness
  checks pass
- soundcheck: proposal-only phase/EQ guard with operator review
- live: shadow/audit-only unless a supervised phase write proof exists

Live exclusions:

- no automatic WING delay writes
- no automatic polarity inversion
- no automatic fader/gain boost as part of phase correction
- no PACE or proprietary Pi logic

## How This Helps EQ

1. Detect likely comb filtering before EQ.
2. Classify whether the issue is broadband level, spectral balance, or phase
   cancellation.
3. If phase is the likely cause, propose a polarity/delay candidate before any
   EQ boost.
4. If phase confidence is low because of bleed, do not apply EQ aggressively.
5. Use low-frequency focus for sources where cancellation is most audible:
   kick/bass, toms, bass DI/mic, guitar DI/mic, overheads/room, playback stems.
6. Feed the result into existing AutoEQ as a guard:
   - `phase_ok`: normal EQ proposal can proceed
   - `phase_risk`: EQ proposal must be conservative
   - `phase_candidate`: show delay/polarity suggestion first
   - `phase_unreliable`: hold action and request longer soundcheck signal

## Current Repo Fit

AutoMixer already has the right building blocks:

- `backend/phase_alignment.py` measures GCC-PHAT delay, phase angle,
  coherence, group delay, Hilbert phase, LMS delay, polarity candidate, and
  bleed risk.
- `tests/test_gcc_phat.py` validates basic GCC-PHAT delay behavior.
- `Docs/AUTO_EQ_INTEGRATION.md` already defines an audit/shadow/proposal-only
  AutoEQ safety boundary.
- `tests/test_source_knowledge.py` already treats drum time alignment as a
  shadow-only source-grounded rule.

The next safe step is not a new plug-in clone. It is to connect existing phase
measurements to AutoEQ decision making through a small guard object.

## Practical Next Step

Add a proposal-only object such as:

```text
PhaseAwareEQGuardResult
```

Fields:

- `group_id`
- `anchor_channel`
- `candidate_channel`
- `phase_mode`
- `low_freq_focus`
- `coherence`
- `delay_ms`
- `polarity_candidate`
- `phase_confidence`
- `eq_action_policy`
- `reason`

Then write focused synthetic tests:

- low-end cancellation blocks EQ boost
- high bleed lowers phase confidence
- high-weight anchor is not moved
- internal-only mode ignores unrelated groups
- live mode returns proposal-only output
