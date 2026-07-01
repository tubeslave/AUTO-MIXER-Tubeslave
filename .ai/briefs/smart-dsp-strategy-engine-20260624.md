# Brief: Smart DSP Strategy Engine

## Short Thesis

Automixer needs a transport-free smart DSP strategy engine that compares several
possible correction methods for the same audible problem and selects the
fastest, safest operator-review proposal for soundcheck and live-assist use.

## What I Understood

The user does not want the project to be locked to the existing per-method DSP
implementations. If a better solution exists, the system should replace,
reroute or supplement the existing method. The current WING boundary still
applies: no live writes, no queue import, and no readiness closure without real
hardware proof.

## Proposed Solution

- Add `backend/smart_dsp_strategy_engine.py`.
- Add `backend/smart_dsp_evidence_extractor.py` so offline audio arrays can
  become measured smart-DSP evidence instead of manually supplied scores only.
- Add `backend/smart_dsp_replay_evaluation.py` and
  `tools/evaluate_smart_dsp_replay.py`.
- Represent audio evidence as a typed analysis input, with an extractor for
  level, true peak, crest factor, LRA proxy, spectral balance, sibilance,
  presence/air deficit, noise/bleed, transient, phase/mono, masking,
  level-plane, FX tail/delay and dynamics-variance features.
- Add transport-free source/role inference for generic or unknown inputs, with
  declared/inferred role evidence, confidence and fail-closed identity guards.
- Treat unconfirmed or low-confidence source identity as its own selected hold
  strategy: `hold_for_source_identity_confirmation` must beat inferred EQ,
  de-esser, dynamics, gate, FX, pan, gain, mute or master candidates until the
  operator confirms patch, label, PFL audio and source role.
- Treat low-confidence audio evidence as its own selected hold strategy:
  `hold_for_audio_evidence_recheck` must beat EQ, de-esser, dynamics, gate,
  FX, pan, gain, mute or master candidates until the capture/evidence is
  repeated or strengthened.
- Require live level, FX, pan and master moves to carry a stepwise change
  guard: no abrupt jumps, max single-step limits and readback after each step
  must be visible in the operator payload.
- Treat any auto-apply request as its own selected hold strategy:
  `hold_for_operator_review_no_auto_apply` must beat normal DSP candidates and
  require operator review, real WING proof and explicit promotion first.
- Treat unsupported modes as their own selected hold strategy:
  `hold_for_supported_mode_selection` must beat normal DSP candidates until
  the operator chooses `soundcheck` or `live`.
- Add a declared-vs-audio source identity guard: if a confirmed label such as
  vocal conflicts with low-end transient or other acoustic evidence, hold all
  DSP strategy selection until the operator rechecks patch, label and source
  role.
- Add musical-role evidence (`lead`, `support`, `low_end_anchor`, `rhythm`,
  `space_fx`, `master_bus`) so the same source family can get different bounded
  DSP parameters.
- Generate multiple strategy candidates for problems such as muddy vocal,
  sibilance, channel dynamics, level-plane balance, masking, noisy drums, wet
  FX, long FX tail/delay smear, unsafe pan width and master overload.
- Keep muddy vocal decisions evidence-led: stable low-mid buildup can use a
  fast static cut, while dynamic low-mid bloom can select dynamic EQ and keep
  multiband as a slower review-only alternative. Low-mid EQ frequency now uses
  a measured mud peak when audio evidence provides one. Dynamic muddy-vocal
  proposals now use measured low-mid bloom peak/rise/recovery to calculate
  threshold, attack and release, falling back to source-family defaults only
  when the peak/envelope is unavailable.
- Keep rumble-dominant muddy decisions evidence-led: when low-end excess is
  stronger than useful low-mid body and the source is not kick/bass/master,
  choose source-role-aware high-pass cleanup before cutting the body band.
- Keep vocal plosive/proximity decisions evidence-led: short low-band bursts
  can choose dynamic HPF control before static HPF, low-mid cuts or manual
  mic-technique review, preserving vocal body between plosives. Dynamic HPF
  proposals now use measured plosive burst peak/contrast/rise/recovery to
  calculate threshold, max low-band reduction, attack and release, falling back
  to score-based values only when burst envelope evidence is unavailable.
- Keep vocal phrase-noise decisions evidence-led: measured noisy gaps between
  phrases can choose a gentle downward expander before hard vocal gating,
  broad EQ dulling or generic compression. Active live performance holds for
  mic/preamp/room-noise review instead of changing expander behavior. Measured
  gap floor, contrast, flatness, median gap length and quiet-frame coverage now
  drive expander threshold, ratio, range, attack, hold, release and hysteresis.
- Keep line-noise/hum decisions evidence-led: measured 50/60 Hz hum and
  harmonics can choose a narrow notch stack before HPF, broad low cuts or
  generic muddy EQ, while still telling the operator to check DI, cable, ground
  lift and power-path causes. Measured harmonic frequencies and prominence now
  drive each notch gain/Q instead of using a fixed harmonic-count preset.
- Keep harsh/sibilant vocal decisions evidence-led: de-essing requires actual
  sibilance evidence, while harsh dynamic vocal peaks can select a bounded
  compressor attack/release envelope adjustment before broad EQ cuts. De-esser
  proposals carry measured sibilance peak and measured band-width evidence into
  `band_start_hz`, `band_end_hz`, `bandwidth_hz` and `q`; measured
  sibilance-event peak/contrast/rise/recovery now drive threshold, range,
  attack and release when available instead of only selecting a default
  frequency or score-based depth.
- Keep source dynamics evidence-led: vocal, bass, guitar and keys should not
  all collapse into one generic compressor recipe. Bass can choose RMS sustain
  leveling that preserves the fundamental, guitar can choose pick/sustain
  compression, and keys can choose slow soft-knee swell leveling, all with no
  makeup-gain headroom bypass and live authority denied. Measured envelope
  peak-to-body, rise time and recovery time now drive threshold, ratio,
  attack, release and target gain-reduction where audio evidence is available.
- When both sibilance and separate presence-band harshness are strong in
  soundcheck, compare and allow a split `de_esser_then_dynamic_presence_control`
  chain to beat either standalone de-essing or a broad presence cut. The
  extractor keeps `presence_harshness_score` separate from sibilance so a
  sibilance-only vocal still chooses the narrower de-esser.
- Keep non-vocal instrument harshness evidence-led: guitar, keys and playback
  sources compare dynamic presence taming, broad presence cut and level-plane
  correction. Explicit pan/width requests and source-level pressure can beat
  tone changes when they are the more direct fix.
- Keep phase/mono risk above tone shaping: when unsafe phase correlation or
  mono-sum loss appears together with harshness, soundcheck should choose
  width/mono compatibility correction first, while live should hold for
  operator review before EQ/dynamic-EQ tone changes.
- Keep dull-source decisions evidence-led: soundcheck may select a bounded
  presence/air EQ lift only when the evidence is not really muddiness or
  harshness, while live holds the boost for operator review and WING proof.
  Backing/support sources use a lower boost cap and preserve the tucked role.
- Keep noisy drum/tom gating evidence-led: compare gentle expander, hard gate,
  clean-hit bleed-aware gate, manual mute review and hold/block behavior, with
  thresholds, ranges, hysteresis and explicit tail/chatter blockers derived
  from evidence. Clean-hit gate proposals now use measured hit peak, hit-to-bleed
  margin and tail-decay timing to calculate threshold, hold, release,
  hysteresis and range instead of relying only on broad bleed/tail risk scores.
  Soundcheck can choose the fastest clean-hit gate, expander or manual mute
  review; live holds first because gate/expander/mute changes can cut real hits
  without operator readback.
- Keep drum transient decisions evidence-led: when snare/kick/drum evidence
  points to attack or peak pressure, compare gentle peak-aware compression,
  manual hold and hard transient clamp review instead of forcing the problem
  through generic channel dynamics or EQ. Transient proposals calculate
  threshold, attack and release from measured peak/rise/decay envelope when
  available, falling back to source-family defaults only when evidence is too
  weak.
- Keep snare/tom ring decisions evidence-led: when evidence points to a narrow
  ringing body resonance, compare dynamic ring-notch control, no-boost static
  notch, tuning/mic/damping review and a blocked broad-de-edge decoy instead of
  treating the problem as cymbal harshness, generic muddy EQ, transient
  compression or gate tuning. Measured ring bandwidth, ring peak and decay time
  now drive dynamic/static notch Q, threshold, max gain reduction, attack and
  release when available, with score-only fallback kept explicit.
- Score candidates on speed, precision, musical usefulness, safety,
  reversibility and operator clarity.
- Select the best safe candidate or fail closed with blockers.
- Add explicit safety-priority arbitration for mixed live problems: high
  feedback risk selects the best unblocked feedback-risk reduction candidate
  before de-essing, tone shaping or other musical correction, even when the
  musical correction has a higher weighted score.
- Treat stale context, missing current-state readback, missing rollback plan
  and live-authority-tainted context as their own selected hold strategy:
  choose `hold_for_context_readback_or_rollback_recheck` before any EQ,
  dynamics, gate, FX, pan, gain, mute or master candidate, while preserving
  those DSP options as alternatives for later review.
- Treat routing/scene/snapshot intent as a first-class safety guard: choose
  `hold_for_routing_scene_snapshot_review` before any EQ, dynamics, gate, FX,
  pan, gain, mute or master candidate when operator intent or problem hints ask
  for routing, scene or snapshot behavior.
- Treat live master/output intent as its own guard even when source evidence
  also detects a channel problem: choose
  `hold_for_live_master_output_intent_review` before channel EQ, dynamics, FX,
  pan or gain so PA-level moves cannot be hidden behind a normal DSP proposal.
- Treat live EQ intent as its own guard even when source evidence also detects
  a tone problem: choose `hold_for_live_eq_intent_review` before channel EQ,
  dynamics, FX, pan or gain so EQ boosts/cuts/notches/filters require operator
  review, EQ-band ownership checks, readback, rollback and real WING proof.
- Treat live source gain-up intent as its own guard even when source evidence
  also detects a tone problem: choose
  `hold_for_live_source_gain_up_intent_review` before channel EQ, dynamics,
  FX, pan or gain so level-increase requests require operator review, headroom,
  feedback, readback, rollback and real WING proof.
- Treat live compressor/gate/mute/FX processing intent as its own guard even
  when source evidence also detects a tone problem: choose
  `hold_for_live_processing_intent_review` before channel EQ, dynamics, gate,
  mute, FX, sidechain or ducking so aggressive live processing requests require
  operator review, readback, rollback and real WING proof.
- Treat live pan/width intent as its own guard even when source evidence also
  detects a tone problem: choose `hold_for_live_pan_width_intent_review` before
  channel EQ, dynamics, pan or width so spatial moves require operator review,
  source identity, phase/mono evidence, readback, rollback and real WING proof.
- Explain the selection as structured `selected_rationale` plus
  `alternative_tradeoffs`, including blockers and score dimensions that made
  competing strategies worse.
- Emit machine-checkable `operator_explanation_sections` for every proposal:
  problem found, selected processing type, winning strategy, why it was chosen,
  alternatives that lost, proposed parameters, remaining risk, operator
  verification and rollback/readback requirements.
- Emit method-specific future WING readback contracts for every candidate:
  EQ/dynamic-EQ must use a frequency-matched or free band and must not
  overwrite feedback notches or operator-locked bands; dynamics, gate/expander,
  FX, pan/width, gain/fader/master and mute proposals each require their own
  before/after readback fields, preserve the `0 dBFS` fader ceiling and keep
  routing/scene/snapshot writes forbidden.
- Treat gain staging / Ayaic level-plane balance as a first-class problem:
  soundcheck may propose bounded gain-up with headroom/feedback/bleed checks,
  while live prefers gain-down and holds unsafe gain-up.
- Treat quiet muddy/dull soundcheck sources as level-plane problems before
  tone problems when evidence is safe: if the channel is below its Ayaic plane,
  has headroom, low bleed and low feedback risk, prefer bounded level restore
  before low-mid or presence EQ; if the summed bus is already hot, prefer bus
  rebalance before adding source gain or carving tone.
- Measure summed bus/group impact for level-plane decisions: when a source is
  quiet but the summed bus is already hot or above its Ayaic plane, block the
  source gain-up candidate and prefer a bus rebalance proposal before adding
  channel gain.
- Treat group/stem buses as first-class targets: when the bus itself is hot,
  too quiet, tonally heavy or dynamically unstable, compare group trim,
  bounded soundcheck group lift, upstream member rebalance, group compression
  review, group EQ review and live hold behavior. Safe soundcheck group
  dynamics can choose bounded no-makeup bus compression, while hot/headroom-risk
  groups still trim first and live stays review-held.
- Treat kick/bass conflicts as pair decisions instead of generic masking:
  compare no-boost pocket EQ split, source-keyed dynamic pocket ducking,
  level-plane rebalance and arrangement-aware hold. Sustained register overlap
  can now select the arrangement hold before EQ or ducking, while rhythmic
  overlap can still select dynamic pocket ducking.
- Treat solo low-end anchor resonance as its own problem instead of folding it
  into generic muddy EQ or channel dynamics: bass/kick evidence can compare
  dynamic low-end resonance control, no-boost static pocket cut, level-plane
  rebalance and hold-for-PA/sub-bus context. Soundcheck prefers dynamic control
  when it is the precise fast fix; live prefers the conservative static cut and
  blocks dynamic EQ for operator review. Measured resonance frequency,
  bandwidth, peak and sustain now drive Q, threshold, max gain reduction,
  attack and release when available.
- Treat FX intelligibility as more than wet-level: compare reduce send/return,
  decay/filter cleanup, return ducking and delay-feedback cleanup. When delay
  smear dominates over wetness/decay risk, choose delay-return ducking and
  feedback reduction instead of just lowering the whole FX return.
- Treat FX wash as a possible root cause of perceived muddy vocals: when
  soundcheck evidence shows strong FX wetness/decay/delay pressure but low
  low-end excess, safe headroom and no level-plane issue, choose FX cleanup
  before low-mid channel EQ even if the EQ candidate has a slightly higher
  weighted score.
- Keep live FX more conservative than soundcheck: for delay smear, long tail or
  wet-level problems during live mode, choose a reversible send/return pullback
  before ducking, decay or delay-time edits; keep those precise moves as
  review alternatives.
- Treat extreme FX monitor-feedback risk as a safety problem first: select a
  send/return pullback and block ducking/decay/delay edits until monitor sends
  and FX-return headroom are checked.
- Treat source feedback risk as its own strategy family: live can choose a
  bounded gain/fader pullback, compare a narrow feedback-notch review candidate,
  and still block unsafe boosts without granting live authority.
- Measure narrow feedback ring peaks from offline audio, carry frequency,
  confidence and source into strategy search, and let soundcheck choose a
  measured narrow notch when that is more precise than a broad gain pullback.
  Sibilance-only peaks, FX returns and group/stem buses stay in their own
  branches instead of being misrouted through source feedback.
- Treat master/output overload as a master-bus decision, not generic channel
  EQ: compare true-peak trim, limiter ceiling guard, broad low-end EQ cleanup
  broad presence harshness cleanup and gentle bus compression. Soundcheck can
  choose compression when dynamics pressure is primary and true-peak headroom
  is safe. Bus-compression proposals now use measured master dynamics
  peak-to-median, rise and recovery evidence to calculate threshold, ratio,
  attack, release and target gain reduction when available; live master/output
  tone or dynamics cases hold for operator review, while peak-risk cases stay
  trim-first.
- Size master trim from both measured dBTP overage and true-peak headroom
  deficit: low headroom can require a larger trim even when
  `master_true_peak_db` is already below the nominal peak threshold, and that
  headroom restoration must beat limiter, broad EQ and bus compression.
- Measure FX-return tail decay and delay-repeat timing from offline audio so
  FX cleanup can calculate `decay_s`, measured tail evidence and delay ducking
  timing instead of relying only on generic wetness scores.
- Treat PA/reference-feed identity as hard evidence: master/reference/capture
  feeds hold analysis-only instead of proposing trim, limiter, EQ or bus
  compression until the PA output identity is confirmed.
- Treat pan/stereo placement as a bounded target/delta decision: soundcheck can
  propose a phase-safe pan target with current pan, target pan, pan delta and
  bounded delta. The target can come from an operator request or measured L/R
  stereo-balance evidence, while live non-center pan changes hold for operator
  review and real WING readback/rollback proof.
- Treat unsafe phase/mono evidence as its own problem even without a pan-width
  request: compare mono-compatible width reduction, polarity/delay alignment
  review and live hold behavior before making tone or placement changes.
- Add vocal-intelligibility masking search for cases where guitar/keys/playback
  or another vocal masks the lead vocal presence band: compare source-keyed
  dynamic EQ, static no-boost pocket EQ, vocal boost and level-plane rebalance,
  preferring dynamic unmasking at soundcheck and static/held behavior in live.
  When reference audio provides a masking overlap band, the dynamic-unmasking
  proposal now derives Q, max gain reduction, threshold, attack and release
  from measured bandwidth and peak-share evidence instead of fixed values.
- Size kick/bass dynamic pocket ducking from measured masking overlap evidence:
  overlap bandwidth and peak share now drive Q, max gain reduction, threshold,
  attack and release, while the old score-only fallback remains visible as
  `masking_score_fallback` when audio evidence is incomplete.
- Add drum/cymbal/overhead harshness search: compare dynamic de-edging, static
  no-boost air/presence cut and mic/level review, while avoiding false
  vocal-sibilance identity holds for declared overhead/cymbal sources. Measured
  harsh-event frequency, bandwidth, peak level, contrast and recovery now drive
  dynamic de-edge Q, threshold, max gain reduction, attack/release and the
  conservative live static cut when available.
- Add snare/tom ring-resonance search: compare dynamic notch, static no-boost
  notch, tuning/mic/damping review and broad-de-edge rejection. Soundcheck can
  choose dynamic control; live chooses the static notch and keeps dynamic ring
  control review-only. Ring-notch parameters use measured bandwidth, peak and
  decay time before falling back to source-family defaults.
- Emit an operator proposal with explicit live-authority denial and rollback /
  future readback expectations, preserving the strategy comparison in the
  proposal, evidence payload and replay artifact.
- Evaluate a default replay pack and write offline JSON/Markdown artifacts
  without live transport.

## Likely Files To Touch

- `backend/smart_dsp_strategy_engine.py`
- `backend/smart_dsp_evidence_extractor.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_smart_dsp_strategy_engine.py`
- `tests/test_smart_dsp_evidence_extractor.py`
- `tests/test_smart_dsp_replay_evaluation.py`
- `tools/evaluate_smart_dsp_replay.py`
- `tests/test_processing_proposal_contracts.py`

## Alternatives Considered

- Extend every existing method first. Rejected for this checkpoint because it
  preserves the old method-first shape instead of adding the requested
  strategy-search layer.
- Add a frontend-only review card. Rejected because it would not add DSP
  decision logic.
- Add a live executor path. Rejected because current WING proof does not allow
  it.

## Risks

- The first engine is heuristic, not yet replay-quality optimized.
- Offline audio evidence extraction uses compact deterministic DSP metrics; it
  is not a replacement for real WING readback or calibrated multitrack replay.
- It is not yet wired into a runtime handler or UI panel.
- It increases operator-review card count and requires contract/audit updates.
- Replay evaluation uses curated heuristic scenarios and is not yet calibrated
  against real multitrack recordings.

## Test Plan

- Focused smart DSP scenarios.
- Synthetic audio evidence tests for muddy vocal, sibilant vocal, dynamic
  vocal compression, soundcheck gain-up, live gain-down, noisy tom, phase/mono
  failure, phase-safe pan target/delta, measured stereo-balance pan target,
  measured low-mid bloom envelope, measured drum transient envelope,
  standalone phase/mono width reduction and
  live hold, long FX tail, true-peak master overload, tonal master low-end
  overload, tonal master presence harshness cleanup, master dynamic-range bus
  compression with measured master-envelope parameters, masking, kick/bass
  overlap, sustained kick/bass arrangement overlap, measured muddy EQ frequency,
  measured de-esser sibilance band-width and sibilance-event envelope,
  measured dynamics envelope compression parameters, dynamic muddy vocal cleanup,
  harsh dynamic vocal envelope tuning,
  non-vocal instrument harshness dynamic EQ, split harsh/sibilant vocal chain
  selection, delay-smear FX ducking cleanup, clean tom bleed-aware gate with
  measured hit/tail parameter calculation,
  extreme tom manual mute review,
  dull vocal presence/air lift, source feedback gain pullback, measured feedback ring notch selection,
  feedback blocking unsafe boosts, backing/support role-aware
  presence lift, hot summed-bus rebalance,
  standalone group/stem bus trim, safe soundcheck group compression and live
  hold behavior, drum transient tamer/hold behavior, snare ring dynamic/static
  notch behavior with measured ring parameter propagation,
  bass sustain/guitar pick/keys swell source-specific compression behavior,
  master reference-feed hold, FX monitor-feedback safety
  pullback, rumble-dominant vocal high-pass selection, measured solo low-end
  resonance dynamic/static parameter calculation, vocal plosive dynamic
  high-pass selection with measured burst-envelope parameters,
  vocal phrase-noise downward expander selection with measured gap-envelope
  parameters,
  active-live vocal noise-floor hold behavior, line-noise/hum notch-stack
  selection with measured harmonic gain/Q calculation, vocal-intelligibility dynamic unmasking/static live pocket
  selection, measured masking overlap parameter calculation, drum/cymbal dynamic
  de-edge/static live air cut selection with measured harsh-event parameters,
  measured FX-tail decay and delay-repeat parameter calculation,
  solo bass low-end dynamic resonance/static live pocket-cut selection,
  snare ring dynamic-notch/static live notch selection with measured
  bandwidth/peak/decay parameter calculation,
  live feedback-risk priority over sibilance correction,
  live feedback-risk priority over intent-only EQ/source-gain/processing/pan
  holds without bypassing low-evidence guards,
  source true-peak headroom pullback before EQ, dynamics, masking or transient
  processing when channel headroom is below the safe floor,
  live vocal harshness static no-boost presence cut before dynamic EQ when
  sibilance evidence is low,
  unknown-source inference and declared-vs-audio identity mismatch hold
  behavior.
- Offline replay/evaluation scenario pack.
- Operator explanation and readback-contract tests for selected rationale,
  alternative tradeoffs, required `operator_explanation_sections` and
  method-specific future WING readback sections across replay scenarios.
- Proposal contract tests.
- Project goal capability matrix tests.
- Full backend suite before claiming completion.

## Where Another Agent May Disagree

Another reviewer may prefer integrating the engine into existing EQ/compressor
modules first. The counterargument is that the user's new goal explicitly asks
for a method-independent selector, so the first architectural move should be the
selector itself.
