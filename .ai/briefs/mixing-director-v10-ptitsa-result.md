# Mixing Director v1.0 experiment: Ptitsa

Input: cumulative Editing Workbench v1.0 multitrack.

Milestone constraints:
- faders only for level;
- pan;
- slow section-aware automation;
- no EQ, compression, saturation, reverb, limiting, or reference matching.

Context inference:
- 18_VALERA_VOX inferred as primary vocal from continuity/activity.
- 09_NIKITA_VOX inferred as secondary vocal.
- arrangement density map derived from non-vocal tracks at 0.5 s resolution.
- section-change candidates detected from slow density novelty.

Balance diagnostics:
- approximate vocal/music ratio: -4.38 dB
- approximate kick/bass ratio: +0.98 dB
- Mix Critic: PASS, no automatic rebalance pass required
- final premaster headroom set by one global trim only; peak -3 dBFS
- output loudness about -23.68 LUFS-I

Human evaluation is still required. The level-matched A/B compares:
A = previous processed autonomous edited premaster
B = Mixing Director v1.0 using only faders, pan and slow automation.

Do not promote this mix merely because the critic passed. PASS means no configured balance guardrail failed.
