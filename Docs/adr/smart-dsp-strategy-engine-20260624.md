# ADR: Smart DSP Strategy Engine

## Context

The current correction-method work has strong proposal contracts, but many
methods still behave as method-specific proposal generators. The new project
goal asks Automixer to find the fastest and most accurate soundcheck/live
processing solution even when that means replacing or supplementing an existing
DSP method.

The repository is still blocked on real WING read-only proof. This decision
therefore covers only offline, dry-run, proposal-only logic.

## Options Considered

1. Keep extending each existing method independently.
2. Add a method-independent strategy-search engine.
3. Add a live smart DSP executor.

## Decision

Add `backend/smart_dsp_strategy_engine.py` as a transport-free strategy-search
engine and register it as a proposal-only method. Add
`backend/smart_dsp_evidence_extractor.py` so offline/synthetic audio can be
measured into the selector's evidence shape before a strategy is chosen. Add
`backend/smart_dsp_replay_evaluation.py` plus
`tools/evaluate_smart_dsp_replay.py` so the selector has a replay/evaluation
artifact instead of only isolated unit checks.

## Why This Won

The strategy-search engine directly matches the new goal: it compares EQ,
dynamic EQ, de-essing, gate/expander, FX, pan, gain/fader and master-bus
alternatives for the same evidence packet, scores them, and chooses the safest
operator-review proposal. The evidence extractor closes the first gap between
real audio and the selector by deriving level, true peak, crest factor,
spectral, sibilance, noise/bleed, transient, phase/mono and masking signals
from offline arrays. The selector also includes a channel dynamics branch that
compares gentle compression, manual level riding and blocked peak-clamp style
compression with threshold, ratio, attack, release, knee, target GR and makeup
parameters. This moves the architecture away from method-first-only behavior
without widening live authority.

The evidence extractor now also performs transport-free source/role inference
for generic or unknown inputs. It records declared role, inferred role, inferred
family, confidence and reason. The inference can route offline strategy search
when explicitly allowed, but without that opt-in the source identity guard keeps
the recommendation analysis-only.

Unconfirmed or low-confidence source identity now selects an explicit hold
instead of hiding behind a normal DSP candidate with global blockers. The
selector chooses `hold_for_source_identity_confirmation`, records
`selection_policy: source_identity_confirmation_guard_priority`, and keeps
inferred EQ, de-essing, dynamics, gate, FX, pan, gain, mute and master
candidates visible only as alternatives until patch, label, PFL audio and
source role are confirmed.

Low-confidence audio evidence now also selects an explicit hold instead of
letting a plausible DSP candidate win with weak proof. The selector chooses
`hold_for_audio_evidence_recheck`, records
`selection_policy: audio_evidence_confidence_guard_priority`, and keeps EQ,
de-essing, dynamics, gate, FX, pan, gain, mute and master candidates visible
only as alternatives until capture/evidence is repeated or strengthened.

Live level, FX, pan and master movement proposals now carry a stepwise change
guard. Any live candidate that changes dB or pan/width values records no abrupt
jump behavior, max single-step limits and readback-after-each-step requirements
in the operator payload, while still denying live apply without WING proof.

Auto-apply requests now select an explicit hold instead of a normal DSP
candidate under a global blocker. The selector chooses
`hold_for_operator_review_no_auto_apply`, records
`selection_policy: auto_apply_operator_review_guard_priority`, and keeps EQ,
dynamics, gate, FX, pan, gain and master choices visible only as alternatives
until operator review, real WING proof and explicit promotion exist.

Unsupported modes now also select an explicit hold instead of a normal DSP
candidate under a global blocker. The selector chooses
`hold_for_supported_mode_selection`, records
`selection_policy: unsupported_mode_guard_priority`, and keeps processing
choices visible only as alternatives until the operator chooses `soundcheck` or
`live`.

Declared source identity is now checked against acoustic evidence even when the
operator supplied a confident label. If a channel declared as vocal carries a
strong low-end transient/bass-like signature, the selector emits
`source_identity_mismatch`, adds `source_identity_evidence_conflict`, and
selects `hold_for_source_identity_recheck` before any EQ, compressor, gate, FX
or pan branch can win. The alternative paths to reroute by acoustic family or
trust the declared label remain blocked until operator confirmation.

The selector now also records musical role separately from source family. Roles
such as `lead`, `support`, `low_end_anchor`, `rhythm`, `space_fx` and
`master_bus` let the same DSP family choose different bounded parameters. For
example, a backing vocal with presence deficit still uses the presence-lift
branch, but caps the boost lower and marks the intent as
`preserve_support_tuck`.

Muddy vocal selection is evidence-led rather than a fixed static EQ recipe. The
selector still uses `low_mid_corrective_cut` for stable low-mid buildup, but
dynamic low-mid bloom can now choose `vocal_dynamic_low_mid_de_mud`. A
review-blocked `vocal_low_mid_multiband_control` candidate remains available
when broad low-mid control may be useful but harder to verify quickly. Both the
static and dynamic low-mid paths now prefer a measured `mud_frequency_hz` from
offline audio evidence before falling back to source-family default
frequencies. Dynamic muddy-vocal cleanup also carries measured low-mid bloom
peak, bloom depth, rise time and recovery time into threshold, attack and
release calculation. The payload records `muddy_dynamic_parameter_source` as
`measured_low_mid_bloom` when that envelope is available.

Rumble-dominant muddy selection now has its own fast path. When the evidence
shows strong low-end excess, the source is not kick/bass/master, and level-plane
rebalance is not the primary issue, the selector can choose
`rumble_dominant_high_pass_cleanup` before cutting the useful low-mid body. The
candidate uses source-role-aware HPF frequency, slope, and explicit
`hpf_before_low_mid_cut` evidence so the operator can verify that the removed
energy was spill/rumble rather than intended source weight.

Vocal plosive/proximity selection is now separated from stable muddiness and
steady rumble. The extractor measures short low-band burst contrast as
`plosive_risk_score`, carries a plosive frequency hint into strategy search,
and measures burst peak, contrast, rise time and recovery. When that evidence
is strong, the selector can choose
`vocal_plosive_dynamic_hpf_control`, comparing it against static HPF, low-mid
cut and manual mic-technique review. The selected dynamic HPF proposal carries
threshold, attack, release, max low-band reduction,
`plosive_parameter_source` and measured burst-envelope fields plus
`preserve_vocal_body_between_plosives` evidence so the operator can solve P/B
bursts without thinning the whole vocal by default.

Vocal phrase-noise selection is now separated from normal vocal dynamics. The
extractor measures noisy gaps between phrases using phrase-gap noise floor,
contrast, spectral flatness, median gap length and quiet-frame coverage, so a
quiet tonal phrase does not become a noise floor false positive. When gap noise
evidence is strong, the selector can choose
`vocal_phrase_noise_downward_expander`, comparing it against a blocked hard
vocal gate, a blocked broad EQ decoy, generic compression and mic/preamp
noise-path review. Expander threshold, ratio, range, attack, hold, release and
hysteresis now come from the measured gap envelope when available. Active live
performance selects the analysis-only mic/preamp/room-noise review path before
any expander behavior change.

Line-noise and hum selection is now separated from generic rumble, muddiness
and feedback. The extractor measures 50/60 Hz mains hum, prominence and
harmonic count. When that evidence is strong on vocal, guitar, keys or playback
sources, the selector can choose `mains_hum_notch_stack`, comparing it against
HPF cleanup, a blocked broad low-cut decoy and DI/ground-lift/power-path
review. Each notch now sizes gain and Q from measured harmonic frequency and
prominence when available, falling back to harmonic-count defaults only when
the measured harmonic list is missing. This keeps narrow electrical noise from
being solved by broad tone loss or by the feedback branch unless feedback
evidence is actually present.

The selector now also treats Ayaic level-plane balance as a first-class
strategy family. It compares bounded gain/fader moves against hold/no-tone
alternatives, uses source and summed-bus scope in the parameters, caps future
fader authority at `0 dBFS`, allows soundcheck gain-up only as a bounded
proposal with safety checks, and keeps live gain-up blocked while allowing
conservative gain-down proposals.

Vocal harsh/sibilant selection is evidence-led rather than method-led. A
de-esser candidate is blocked when sibilance evidence is low, so broad
harshness does not automatically become a de-essing proposal. When harshness
tracks dynamic vocal peaks, the selector can choose
`vocal_compressor_envelope_smoothing`, a bounded compressor attack/release
adjustment with no makeup gain, before broad presence EQ cuts. De-esser
proposals now use measured sibilance peak and measured band-width evidence to
calculate band start/end and Q. Measured sibilance-event peak, contrast, rise
time and recovery time now drive threshold, range, attack and release when
available, instead of relying only on score-based depth around a center
frequency.

When soundcheck evidence shows both strong sibilance and separate presence-band
harshness, the selector can choose
`vocal_de_esser_dynamic_presence_combo`, a split
`de_esser_then_dynamic_presence_control` chain with bounded de-esser frequency,
threshold, ratio, max gain reduction and dynamic presence-band max gain
reduction. The extractor now reports `presence_harshness_score` separately from
overall harshness so a sibilance-only vocal remains a narrower de-esser problem
instead of being over-processed by the combo chain.

Non-vocal instrument harshness is now also evidence-led. Guitar, keys and
playback sources compare `instrument_dynamic_presence_tame`,
`instrument_broad_presence_cut` and `instrument_level_check_before_harsh_eq`.
Soundcheck can choose dynamic EQ for intermittent bite, live can prefer a
conservative broad cut, and level-plane pressure or an explicit pan/width
request can block tone changes when balance or placement is the direct fix.

Presence/air lift is now also evidence-led. The extractor computes
`presence_deficit_score` and only auto-tags `dull_presence` when the evidence is
not dominated by muddiness or harshness. Soundcheck can select
`soundcheck_presence_air_lift`, a bounded high-shelf/peaking EQ boost with
true-peak and feedback blockers. Live mode selects
`hold_presence_boost_for_live_safety` and blocks the boost until operator review
and real WING readback/rollback proof exist. Musical role now bounds that lift:
lead sources can get a larger clarity move, while support/backing sources keep a
smaller boost cap and a tucked-role intent.

Noisy drum/tom selection is now also evidence-led. The gate branch compares
gentle expander, clean-hit bleed-aware gate, fast gate, manual mute review and
hold/block behavior. Gate thresholds, ranges and hysteresis carry
noise-floor/bleed/tail/chatter/transient evidence. Clean tom hits can select
`bleed_aware_clean_hit_gate`, hard gates are blocked under tail or chatter
risk, and extreme bleed plus tail plus chatter selects
`manual_tom_mute_between_hits` instead of pretending a gate preset is safe. The
clean-hit gate now also carries measured hit peak, hit-to-bleed margin and tail
decay into threshold, hold, release, range and hysteresis calculation.

Live noisy drum/tom selection now holds before gate, expander or manual mute
changes. The selector can choose
`hold_live_noisy_drum_processing_for_operator`, record
`selection_policy: live_noisy_drum_review_priority`, and keep the exact gate,
expander and mute alternatives visible with live-review blockers. This prevents
the smart DSP engine from treating a live tom bleed problem as permission to
cut real hits mid-performance.

Drum transient selection is now evidence-led instead of being folded into
generic channel dynamics. Snare/kick/drum evidence with high transient density,
transient strength or crest factor can select
`drum_transient_tamer_compression`, a peak-aware compression proposal with
bounded ratio, attack, release, target gain reduction and `0 dB` makeup.
Measured transient peak/rise/decay evidence now drives threshold, attack and
release when available. The payload records `transient_parameter_source` as
`measured_transient_envelope`; source-family defaults remain the fallback. The
same search compares `manual_transient_hold_for_operator` and a blocked
`fast_transient_clamp_review`, so live/low-headroom contexts can hold instead of
flattening attack or widening authority.

Source-specific channel dynamics now keep bass, guitar and keys from falling
through the same generic compressor recipe. Bass dynamics can select
`bass_sustain_leveling_compression` with RMS detection, slower release and
fundamental preservation. Guitar can select
`guitar_pick_sustain_compression`, preserving pick attack while controlling
sustain swings. Keys can select `keys_swell_leveling_compression`, using slow
soft-knee leveling for pads or layered patches. All three keep makeup gain at
`0 dB` in the selected source-specific candidate and still deny live authority.
When offline audio envelope evidence is available, the selector now carries
peak-to-body level, rise time and recovery time into threshold, ratio,
attack/release and target gain-reduction calculation.

Snare/tom ring resonance is now separated from cymbal harshness, generic
muddiness, transient compression and gate tuning. The extractor measures a
narrow ringing body resonance and carries score, frequency, bandwidth, peak
level and decay evidence into strategy search. The selector can choose
`drum_dynamic_ring_notch_control`, compare `drum_static_ring_notch_cut`, hold
for tuning/mic/damping review, and explicitly reject broad drum/cymbal
de-edging when the measured problem is a body ring. Live mode prefers the
no-boost static notch and blocks dynamic ring control for operator review.
When measured bandwidth/peak/decay are available, they drive dynamic/static
notch Q, threshold, max gain reduction, attack and release before fallback
defaults are used.

Summed bus/group evidence is now a measured input to level-plane decisions, not
only a payload label. If an individual source is below its target plane but the
summed bus has low true-peak headroom or is already above its Ayaic plane, the
selector blocks the source gain-up candidate and prefers
`summed_bus_rebalance_before_source_gain`. This keeps the fastest correction
aligned with bus safety instead of blindly boosting the quiet channel.

Quiet soundcheck sources that are also described as muddy or dull now get a
level-first arbitration rule. When source headroom, bleed and feedback evidence
are safe, the selector can choose `soundcheck_bounded_gain_up` before low-mid or
presence EQ and records
`selection_policy: soundcheck_quiet_level_plane_priority`. If the summed bus is
already hot, it chooses `summed_bus_rebalance_before_source_gain` first and
records `selection_policy: soundcheck_quiet_hot_bus_rebalance_priority`. This
keeps "muddy" from becoming an automatic EQ cut when the more direct fix is
level or bus balance.

Group/stem buses are now also first-class processing targets. When a target
looks like a bus, group, stem, DCA or aux and carries its own level, headroom,
tonal or dynamics evidence, the selector can choose
`group_bus_trim_for_headroom`, compare bounded soundcheck group gain-up,
upstream member rebalance, group-bus compression review, group-bus EQ review
and live hold behavior. This covers group impact directly instead of only using
group evidence as a side input to a channel decision. Safe soundcheck group
dynamics can now select bounded no-makeup group-bus compression when headroom is
healthy; hot/headroom-risk groups still trim first, live group compression stays
operator-reviewed, and group/stem targets no longer fall through to generic
channel compression.

FX strategy selection now separates wet-level problems from long-tail and delay
smear problems. The FX branch compares reduce-send/return, decay/filter cleanup,
ducking, delay-feedback cleanup and blocked FX-depth increase. Offline evidence
includes FX wetness, tail persistence, delay smear, return headroom and monitor
feedback risk. Delay-smear-dominant evidence can now select
`duck_delay_return_cleanup`, lowering delay feedback and ducking repeats instead
of treating the problem as a generic wet-level pullback.

FX wash can now beat channel EQ when it is the more likely root cause of a
muddy vocal at soundcheck. If FX wetness/decay/delay evidence dominates the
muddy score while low-end excess, spectral tilt, level-plane error, feedback and
headroom risks are low, the selector chooses `shorten_fx_decay_filter`, records
`selection_policy: soundcheck_fx_wash_cleanup_priority`, and explains why
`low_mid_corrective_cut` was not selected even if its weighted score is slightly
higher.

Live FX arbitration keeps the same candidate search but changes the winning
policy. When live mode sees delay smear, long tail or wet-level FX
intelligibility evidence, the selector chooses the reversible
`reduce_fx_send_or_return` pullback before ducking or decay/time edits. The
recommendation records `selection_policy: live_fx_safety_pullback_priority`,
and the more precise FX alternatives remain visible for operator review.

Extreme FX monitor-feedback risk now takes precedence over ambience tuning. The
selector can choose `fx_monitor_feedback_safety_pullback`, with bounded
send/return reductions, return-mute review metadata and return filtering. Under
that condition, decay editing, ducking and delay-ducking candidates are blocked
until monitor sends and FX-return headroom are checked.

Source feedback risk now has its own strategy branch instead of only a global
stop. High `feedback_risk` can select `feedback_gain_pullback`, a bounded
gain/fader reduction that preserves the no-live-authority boundary. The same
branch compares `narrow_feedback_notch_review`, but blocks it when the ring
frequency is only estimated, and it blocks unsafe boosts through
`feedback_risk_blocks_boost`.

Live safety arbitration now overrides pure weighted-score selection for mixed
problems with high feedback risk. If a live vocal is both sibilant/harsh and
near feedback, the selector chooses the best unblocked feedback-risk reduction
candidate before de-essing or tone shaping. The recommendation records
`selection_policy: live_feedback_risk_priority`, and alternative tradeoffs say
that feedback reduction preceded musical correction.

Source true-peak headroom risk now has its own branch instead of acting only as
a global blocker. If an ordinary source channel is below the safe headroom
floor, the selector chooses `source_true_peak_headroom_pullback`, calculates a
bounded negative gain/fader delta and records
`selection_policy: true_peak_headroom_safety_priority`. Tone, dynamics,
masking and transient candidates stay visible as alternatives until headroom is
restored; source identity, audio evidence, context, routing/scene/snapshot and
auto-apply guards still win when present.

Live vocal harshness now has a conservative static-EQ arbitration path. When
the vocal is harsh but sibilance evidence is low, live mode selects
`broad_presence_cut`, records
`selection_policy: live_static_vocal_harshness_cut_priority`, and keeps
`presence_dynamic_eq_cut` as an operator-review alternative instead of letting
weighted scoring prefer dynamic EQ during an active performance.

Phase/mono safety arbitration also overrides pure weighted-score selection for
mixed stereo/tone problems. If unsafe phase correlation or mono-sum loss appears
together with harshness, soundcheck chooses `mono_compatibility_width_reduction`
before tone shaping, while live chooses `hold_phase_mono_processing_for_operator`
before EQ or dynamic-EQ changes. The recommendation records
`selection_policy: soundcheck_phase_mono_safety_priority` or
`selection_policy: live_phase_mono_safety_priority`, and alternative tradeoffs
explain that mono compatibility preceded tone or width changes.

Feedback frequency evidence is now measured directly from offline audio when a
narrow peak has enough feedback context. The extractor carries
`feedback_frequency_hz`, `feedback_frequency_confidence` and
`feedback_frequency_source`; the selector can let soundcheck choose
`narrow_feedback_notch_review` over broad gain pullback when the ring frequency
is measured. Sibilance-only peaks without monitor context, FX returns and
group/stem buses stay routed to their own DSP branches so the feedback branch
does not steal unrelated problems.

Kick/bass masking now uses a low-end pair branch instead of the generic masking
path alone. It compares no-boost pocket EQ split, source-keyed dynamic pocket
ducking, level-plane rebalance and arrangement-aware hold. Strong soundcheck
overlap can select dynamic pocket ducking when the level plane is sane; live
mode prefers the no-boost EQ split and blocks the dynamic move for operator
review. When reference audio provides measured masking overlap bandwidth and
peak-share evidence, the dynamic pocket proposal now derives Q, max gain
reduction, threshold, attack and release from that overlap band instead of a
fixed ducking recipe.

Vocal-intelligibility masking now uses the same evidence path for presence-band
unmasking. Source-keyed dynamic EQ can calculate Q, max gain reduction,
threshold, attack and release from the measured masking overlap band; the
payload records `masking_parameter_source`, `masking_bandwidth_hz` and
`masking_overlap_peak_share` so replay and operator review can distinguish
measured parameters from score-only fallback behavior.

Kick/bass arrangement overlap is now a distinct evidence path. When sustained
low-register overlap is stronger than groove/transient evidence, the selector
can choose `arrangement_aware_low_end_separation_hold` before EQ or ducking.
That keeps non-DSP musical problems from being hidden behind a fast carve or a
sidechain proposal.

Solo low-end anchor resonance is now separate from both generic muddiness and
kick/bass masking. Bass/kick evidence with strong low-end excess can compare
`low_end_dynamic_resonance_control`, `low_end_static_pocket_cut`,
`low_end_level_plane_rebalance_before_eq` and
`hold_low_end_resonance_for_context`. Soundcheck can prefer dynamic EQ when the
resonance is the precise fast fix; live prefers a no-boost static pocket cut
and blocks dynamic low-end EQ until operator review, readback and rollback
proof exist. The extractor now measures low-end resonance frequency, bandwidth,
peak level and sustain; the selector uses that evidence to calculate Q,
threshold, max gain reduction, attack and release before falling back to
score-based defaults.

Master/output selection now routes master evidence through master-bus candidates
instead of generic channel EQ. True-peak overload selects bounded trim first,
limiter ceiling remains a PA/reference-feed safety-checked alternative, tonal
low-end overload with safe headroom selects broad low-shelf cleanup, presence
harshness with safe headroom selects broad peaking cleanup, and bus compression
can be selected as a bounded soundcheck proposal when dynamics pressure is the
primary issue and true-peak headroom is safe. Bus-compression parameters now
use measured master dynamics peak-to-median, rise time and recovery time when
available, so threshold, ratio, attack, release and target gain reduction are
derived from envelope evidence instead of only from a generic pressure score.
Live bus compression and true-peak-risk cases remain blocked for operator
review or trim-first handling.

Master trim sizing now uses both master dBTP overage and the measured
true-peak-headroom deficit. A PA feed with safe-looking `master_true_peak_db` but
only `0.2 dB` headroom now proposes enough trim to restore the `1.5 dB` target
headroom, records `selection_policy: master_true_peak_headroom_priority`, and
keeps limiter, broad EQ and bus compression visible as losing alternatives until
headroom is restored.

Live master/output tone and dynamics now use an explicit hold-first strategy
when true peak is already safe. The selector can choose
`hold_live_master_output_processing_for_operator` before master EQ,
bus-compression or limiter changes, record
`selection_policy: live_master_output_review_priority`, and keep the exact
master-bus DSP alternatives visible for operator review.

Master-bus processing now also checks feed identity. Source roles or labels that
look like reference, capture, recording, broadcast, stream or analyzer feeds
select `hold_master_reference_feed_no_pa_write` and remain analysis-only. This
prevents true-peak or tonal evidence from turning a non-PA reference feed into a
master trim, limiter, bus-compression or EQ proposal.

Pan/stereo selection now treats placement as a bounded target/delta decision,
not a generic width flag. Soundcheck phase-safe pan requests can select
`phase_safe_pan_target` with target pan, current pan, pan delta, bounded delta,
phase correlation and mono-sum evidence. Phase-safe soundcheck stereo imbalance
can also calculate the target from measured L/R balance, recording
`pan_target_source: measured_stereo_balance`, measured balance confidence and
measured pan delta. Live non-center pan changes keep the safe hold candidate
selected and block the pan move with
`live_pan_change_requires_operator_review_and_wing_proof` until real WING
readback/rollback proof exists.

Phase/mono safety now also works without an explicit width request. If stereo
evidence shows unsafe phase correlation or mono-sum loss, the selector can
choose `mono_compatibility_width_reduction`, compare a blocked
`phase_polarity_alignment_review`, or select
`hold_phase_mono_processing_for_operator` in live mode. This makes phase/mono
evidence a real DSP decision branch instead of only a pan-request blocker.

Operator explanation is now backed by structured comparison data instead of only
a sentence of scores. Each recommendation records `selected_rationale` and
`alternative_tradeoffs`, including candidate blockers, score gaps and
`why_not_selected` reasons. The same comparison is emitted into the operator
proposal and replay artifact so "why this strategy beat the alternatives" is
machine-checkable.

Operator explanation now also has a required section contract. Each proposal
emits `operator_explanation_sections` with the problem found, selected
processing type, winning strategy, why it was selected, alternatives that lost,
proposed parameters, remaining risk, operator verification and
rollback/readback requirements. Replay evaluation fails when these fields are
missing or drift away from the selected strategy.

Context safety gaps are now selected as an explicit strategy instead of being
hidden behind a normal DSP candidate with global blockers. If context freshness
is stale, current-state readback is missing, rollback planning is missing, or a
fresh context carries live-authority/write/query flags, the selector chooses
`hold_for_context_readback_or_rollback_recheck` and records
`selection_policy: context_readback_rollback_guard_priority`. EQ, dynamics,
gate, FX, pan, gain, mute and master candidates remain visible only as
alternatives until read-only context, readback and rollback are proven.

Routing, scene and snapshot intent is now also selected as an explicit hold
strategy instead of relying only on the future readback contract. If the
operator intent or problem hints include routing, scene recall/save/load or
snapshot behavior, the selector chooses
`hold_for_routing_scene_snapshot_review`, records
`selection_policy: routing_scene_snapshot_guard_priority`, and leaves EQ,
dynamics, gate, FX, pan, gain, mute and master candidates as alternatives only.

Live master/output intent is now guarded at the intent layer, not only when the
target source is already classified as a master bus. If the live operator
intent or problem hints ask for master/output movement while channel evidence
also suggests EQ or dynamics, the selector chooses
`hold_for_live_master_output_intent_review`, records
`selection_policy: live_master_output_intent_guard_priority`, and keeps channel
DSP candidates as alternatives until PA-feed identity, readback, rollback and
real WING proof are available.

Live EQ intent is now guarded before generic source gain-up intent. If the live
operator intent asks for EQ, highs, air, presence, cuts, notches or filters
while channel evidence also suggests a normal tone correction, the selector
chooses `hold_for_live_eq_intent_review`, records
`selection_policy: live_eq_intent_guard_priority`, and keeps normal DSP
candidates as alternatives until source identity, EQ-band ownership, feedback
notch safety, readback, rollback and real WING proof are available. This keeps
phrases like `boost highs` from being misclassified as channel gain-up.

Live source gain-up intent is now guarded the same way. If the live operator
intent asks to boost, raise, increase, push, turn up, lift, make louder, or
otherwise move channel gain/fader/trim upward while channel evidence also
suggests EQ or dynamics, the selector chooses
`hold_for_live_source_gain_up_intent_review`, records
`selection_policy: live_source_gain_up_intent_guard_priority`, and keeps channel
DSP candidates as alternatives until source identity, headroom, feedback,
readback, rollback and real WING proof are available.

Live compressor, gate, mute, FX, sidechain and ducking intent is now guarded at
the same intent layer. If the live operator intent asks to slam compression,
hard-gate, mute, add more reverb/delay/FX, sidechain or duck while channel
evidence also suggests EQ or dynamics, the selector chooses
`hold_for_live_processing_intent_review`, records
`selection_policy: live_processing_intent_guard_priority`, and keeps normal DSP
candidates as alternatives until source identity, readback, rollback and real
WING proof are available.

Live pan and stereo-width intent is now guarded at the same intent layer. If
the live operator intent asks to pan, move left/right, widen, narrow or spread
a source while channel evidence also suggests EQ or dynamics, the selector
chooses `hold_for_live_pan_width_intent_review`, records
`selection_policy: live_pan_width_intent_guard_priority`, and keeps normal DSP
candidates as alternatives until source identity, phase/mono evidence,
readback, rollback and real WING proof are available.

High live feedback risk now outranks intent-only live holds. If the only global
blocker is a live EQ, source-gain, processing or pan/width intent review, the
selector chooses the bounded `feedback_gain_pullback`, records
`selection_policy: live_feedback_risk_priority`, and leaves the intent hold as
an alternative explaining that risk reduction precedes intent review. Hard
source identity, audio evidence, context, routing/scene/snapshot and
auto-apply guards still win when present.

Future WING readback contracts are now method-specific. The selector records
common safety invariants for every candidate, including real WING proof,
operator approval, pre-promotion dry-run, `0 dBFS` fader ceiling and no
routing/scene/snapshot writes. EQ and dynamic-EQ candidates also require a
frequency-matched or free band and must not overwrite feedback notches or
operator-locked bands. Dynamics, gate/expander, FX, pan/width,
gain/fader/master and mute candidates each carry their own before/after
readback sections, and replay evaluation fails when the selected strategy is
missing the required section.

## Rejected Alternatives

- Per-method-only expansion was rejected for this checkpoint because it would
  not create the requested cross-method selector.
- Live execution was rejected because M001 still waits for external WING
  read-only validation and non-fader/gain live writes remain outside the proven
  authority boundary.

## Implementation Plan

- Add typed smart DSP analysis and recommendation objects.
- Add typed offline audio evidence extraction and conversion into
  `SmartDSPAnalysisInput`.
- Add source/role inference metadata and an explicit opt-in path for using
  high-confidence inferred identity in offline strategy search.
- Add declared-vs-audio identity mismatch evidence and a fail-closed hold path
  that prevents wrong-family DSP decisions until patch/source identity is
  rechecked.
- Add musical-role evidence and carry it into recommendation, proposal evidence
  and replay results.
- Generate alternative candidates from soundcheck/live evidence.
- Add muddy vocal dynamic EQ and review-only multiband candidates so dynamic
  low-mid bloom does not always become a static EQ cut, with measured bloom
  peak/rise/recovery driving dynamic EQ threshold, attack and release.
- Add rumble-dominant HPF selection so strong low-end spill can choose
  high-pass cleanup before low-mid body cuts.
- Add vocal plosive dynamic HPF selection so short low-band bursts can be
  measured into burst peak/contrast/rise/recovery parameters and controlled
  without turning every proximity issue into a static HPF or low-mid cut.
- Add vocal phrase-noise evidence and expander selection so noisy gaps between
  phrases can choose a gentle downward expander before hard gating, broad EQ or
  generic compression, while active live performance holds for mic/preamp/noise
  review.
- Add measured vocal phrase-gap duration and coverage so expander threshold,
  ratio, range, attack, hold, release and hysteresis are calculated from the gap
  envelope when available.
- Add line-noise/hum evidence and notch-stack selection so measured 50/60 Hz
  mains noise can choose narrow EQ before HPF, broad low-cut or generic muddy
  EQ, while keeping DI/cable/ground-lift/power review in the operator check.
- Add measured line-noise harmonic gain/Q calculation so each notch follows
  harmonic prominence instead of a fixed stack preset.
- Add channel dynamics candidates for high crest/LRA vocal, bass, guitar and
  keys evidence, with source-specific bass sustain leveling, guitar
  pick/sustain compression and keys swell leveling before the generic
  compressor, while keeping makeup bounded or zero and denying live apply
  authority.
- Add measured dynamics envelope extraction so compressor threshold, ratio,
  attack, release and target gain reduction can be calculated from peak-to-body
  level, rise time and recovery time instead of fixed family defaults.
- Add harsh dynamic vocal envelope tuning so compressor attack/release can beat
  de-essing or broad EQ when the evidence points to dynamic peaks rather than
  sibilance.
- Add measured muddy-EQ frequency extraction so static and dynamic low-mid
  cleanup can use the actual low-mid peak instead of defaulting to a generic
  vocal/source-family frequency.
- Add measured de-esser band-width and sibilance-event envelope extraction so
  sibilance proposals carry band start/end, Q, threshold, range, attack and
  release parameters from audio evidence.
- Add split harsh/sibilant vocal chain selection so soundcheck can combine a
  de-esser with dynamic presence control only when both sibilance evidence and
  separate presence-band harshness evidence are present.
- Add non-vocal instrument harshness candidates so guitar, keys and playback
  harshness can choose dynamic EQ, broad EQ, or level-plane correction instead
  of falling through to a generic hold.
- Add presence/air deficit evidence and bounded soundcheck EQ-boost candidates
  with live hold behavior, true-peak blockers, feedback blockers and role-aware
  support-source boost caps.
- Add standalone level-plane/gain-staging candidates for channels and buses so
  obvious balance problems do not masquerade as EQ or compressor problems.
- Add summed bus/group evidence fields and blockers so source gain-up loses to
  bus rebalance when the group or stem is already hot.
- Add standalone group/stem bus target candidates so a hot or quiet group bus
  can choose group trim, bounded soundcheck lift, member rebalance review, bus
  compression review, bus EQ review or live hold without pretending it is an
  ordinary source channel. Keep safe soundcheck group compression available as
  a bounded no-makeup proposal, but route group/stem dynamics away from generic
  channel compression.
- Add noisy tom/drum clean-hit gate, manual mute review and blocking candidates
  so stable hits can beat expander tuning while unstable bleed/tail/chatter
  evidence can still beat gate or expander tuning.
- Add measured gate hit/tail parameter extraction so clean-hit gate proposals
  calculate threshold, hold, release, range and hysteresis from hit peak,
  hit-to-bleed margin and tail decay.
- Add drum transient-control candidates so high attack/peak evidence can choose
  gentle peak-aware compression, manual hold or blocked hard clamp review
  instead of being forced through generic EQ/dynamics, with measured
  peak/rise/decay driving threshold, attack and release when available.
- Add snare/tom ring-resonance evidence and candidates so narrow drum body
  ringing can choose dynamic notch control at soundcheck, conservative no-boost
  static notch in live, tuning/mic/damping review, or a blocked broad-de-edge
  decoy instead of being misrouted through cymbal harshness, generic muddy EQ,
  transient compression or gate tuning.
- Add measured drum-ring bandwidth, peak level and decay-time extraction so
  ring-notch Q, threshold, max gain reduction, attack and release are calculated
  from the resonance envelope when available.
- Add FX decay/filter and delay-ducking candidates so wet FX problems are not
  always reduced to send/return level changes, and make delay-ducking win when
  delay smear is the dominant intelligibility problem.
- Add FX monitor-feedback safety pullback so high monitor risk blocks complex
  ambience edits and chooses the fastest reversible risk reduction first.
- Add source feedback-risk candidates so live/source feedback can choose a
  bounded gain pullback, compare a narrow notch review, and block unsafe boosts
  without widening live authority.
- Add live feedback-risk priority arbitration so mixed feedback plus
  sibilance/harshness evidence chooses bounded risk reduction before musical
  correction, even when the musical correction has a higher weighted score.
- Add live feedback-risk priority arbitration over intent-only live EQ,
  source-gain, processing and pan/width holds without bypassing hard
  source/context/evidence guards.
- Add source true-peak headroom-risk candidates so ordinary channels can choose
  bounded gain/fader pullback before tone, dynamics, masking or transient
  processing when headroom is below the safe floor.
- Add live vocal harshness arbitration so low-sibilance harsh vocals prefer a
  no-boost static presence cut before dynamic EQ in active performance.
- Add measured feedback ring-frequency evidence so soundcheck can select a
  precise narrow notch when the frequency is known, while sibilance, FX and
  group/stem false positives are kept out of source-feedback selection.
- Add kick/bass-specific masking candidates that use masking strength, measured
  overlap bandwidth, peak share and low-end pocket frequency instead of a
  generic EQ/ducking choice, including an arrangement-overlap evidence score
  that can prefer a no-DSP hold before EQ or sidechain ducking.
- Add solo low-end anchor resonance candidates so bass/kick boom can choose
  dynamic low-end EQ at soundcheck, conservative no-boost static pocket cut in
  live, level-plane rebalance review or PA/sub-bus hold instead of falling
  through to generic muddy EQ or channel dynamics, with measured
  frequency/bandwidth/peak/sustain driving dynamic/static EQ parameters.
- Add vocal-intelligibility masking candidates that compare source-keyed
  dynamic EQ, static no-boost pocket EQ, vocal boost and level-plane rebalance,
  preferring dynamic unmasking at soundcheck and static/held behavior in live,
  with measured masking-overlap parameters carried into replay and operator
  proposals.
- Add drum/cymbal/overhead harshness candidates that compare dynamic de-edging,
  static no-boost air/presence cuts and mic/level review, while keeping live
  dynamic de-edging behind operator review.
- Add measured drum/cymbal harsh-event extraction so frequency, bandwidth, peak
  level, contrast and recovery drive de-edge Q, threshold, max gain reduction,
  attack/release and static live cut parameters when available.
- Add master/output-specific candidates for true-peak trim, limiter ceiling,
  broad low-end EQ cleanup, broad presence harshness EQ cleanup and
  bounded soundcheck bus compression for dynamics pressure, with measured
  master-envelope parameters when evidence is available.
- Add a master PA/reference-feed identity guard so reference/capture feeds fail
  closed before any master-bus processing candidate is proposed.
- Add bounded pan/stereo target and delta parameters so soundcheck can propose
  phase-safe placement changes, while live pan changes stay operator-reviewed
  and blocked from live authority.
- Add standalone phase/mono-risk candidates so unsafe stereo evidence can
  choose mono-compatible width reduction, polarity/delay alignment review or
  live hold even when no pan-width request was made.
- Add measured FX-tail and delay-repeat evidence so FX candidates can calculate
  decay reduction and delay-ducking timing from offline audio instead of
  relying only on generic wetness or decay scores.
- Score candidates by speed, precision, musical usefulness, safety,
  reversibility and operator clarity.
- Emit selected rationale and alternative tradeoffs into recommendation,
  proposal evidence and replay results so operator explanations identify both
  the winning strategy and why competing strategies lost.
- Emit required `operator_explanation_sections` into recommendation, proposal
  evidence and replay results so the operator-facing explanation answers the
  goal checklist as structured data, not only as prose.
- Emit method-specific future WING readback contracts into every candidate and
  replay result, including EQ-band protection, dynamics/gate/FX/pan/gain/mute
  before/after readback, `0 dBFS` fader ceiling and routing/scene/snapshot
  denial.
- Fail closed on stale context, live-authority-tainted context, missing
  readback, missing rollback, low headroom, feedback risk and missing source
  identity.
- Emit operator-readable explanations, rollback expectations and future WING
  readback contracts.
- Register the method as proposal-only, not executor-capable.
- Evaluate a default scenario pack covering muddy vocal, sibilant vocal,
  dynamic muddy vocal, dull vocal presence lift, backing/support role-aware
  presence lift, live dull-vocal boost hold, live vocal feedback gain pullback,
  feedback blocking unsafe presence boost, dynamic vocal compression,
  bass sustain leveling compression, live guitar pick/sustain compression,
  soundcheck gain-up, live gain-down, kick/bass masking with low headroom,
  live source true-peak headroom pullback before EQ,
  live vocal static harshness cut before dynamic EQ,
  soundcheck dynamic pocket ducking, live conservative EQ split, solo bass
  low-end dynamic resonance/static live pocket cut, snare ring dynamic/static
  notch with measured ring parameters, live feedback-priority over sibilance
  correction, noisy tom, FX
  monitor risk, long FX tail cleanup with measured decay, pan phase/mono failure,
  soundcheck delay-smear FX ducking cleanup with measured repeat timing, phase-safe pan delta,
  measured stereo-balance pan target, live pan-hold review,
  standalone soundcheck phase/mono width reduction, standalone live phase/mono hold, true-peak master overload,
  tonal master low-end cleanup, tonal master presence harshness cleanup, master
  dynamic-range bus compression, harsh dynamic vocal envelope tuning,
  non-vocal instrument harshness dynamic EQ,
  split harsh/sibilant vocal chain selection, vocal phrase-noise expander
  selection, line-noise/hum notch-stack selection, clean tom bleed-aware gate,
  extreme noisy tom manual mute review, soundcheck snare transient tamer with
  measured transient envelope, live low-headroom transient hold,
  FX monitor-feedback safety pullback, master
  reference-feed hold, inferred source identity guard, declared-vs-audio
  identity mismatch hold, measured feedback ring notch, hot summed-bus
  rebalance, standalone group bus trim, soundcheck group compression,
  measured muddy EQ frequency, live quiet group gain hold, stale
  context, vocal-intelligibility dynamic unmasking/static live pocket and
  drum/cymbal dynamic de-edge/static live air cut with measured harsh-event
  parameters, and
  live-authority-tainted context.
- Add synthetic audio tests proving the extractor can drive strategy selection
  for muddy vocal, sibilance, channel compression, level-plane moves, noisy
  tom, drum transient control, measured FX tail cleanup, measured FX delay-smear cleanup, masking, phase/mono,
  standalone phase/mono risk, phase-safe pan delta, standalone group bus trim,
  master overload, master presence harshness, non-vocal instrument harshness
  vocal phrase-noise expander selection, vocal masking/unmasking and
  drum/cymbal harshness with measured harsh-event parameters and without false
  vocal-identity holds, plus
  snare ring-resonance dynamic/static notch selection with measured
  bandwidth/peak/decay propagation,
  unknown-source inference cases without transport access.
- Write `summary.json` and `summary.md` artifacts under
  `artifacts/smart_dsp_replay_evaluation/`.

## Test Plan

- Focused tests for muddy vocal, sibilant vocal, measured de-esser band-width
  and measured sibilance-event envelope,
  kick/bass masking with measured overlap parameter calculation,
  sustained kick/bass arrangement overlap, noisy tom,
  dynamic vocal compression, measured muddy EQ frequency, level-plane gain-up/down, FX monitor risk, FX
  decay/filter cleanup, pan mono/phase failure, phase-safe pan target/delta,
  measured stereo-balance pan target, live pan-hold review,
  standalone phase/mono risk, true-peak master overload,
  master low-end cleanup, master presence harshness cleanup, master dynamic-range
  bus compression with measured envelope parameters and stale/live
  authority context.
- Evidence extractor tests for audio-derived smart-DSP inputs and selected
  strategies, including dynamic muddy vocal cleanup, dynamic guitar
  pick/sustain compression, harsh dynamic vocal
  envelope tuning, split harsh/sibilant vocal chain selection, dull vocal
  presence/air lift and a synthetic
  quiet-source/hot-bus conflict plus clean tom bleed-aware gate and extreme
  noisy tom manual mute review, standalone group bus trim,
  drum transient control with measured envelope, measured FX-tail/delay parameter
  calculation, FX monitor-feedback safety pullback, master reference-feed hold, measured feedback ring notch
  behavior, master dynamic-range bus compression, sustained kick/bass
  arrangement-overlap hold behavior, measured masking overlap parameter
  propagation, measured plosive burst-envelope parameter propagation,
  measured low-end resonance parameter propagation,
  vocal phrase-noise expander selection with measured gap-envelope propagation,
  line-noise/hum notch-stack selection with measured harmonic gain/Q propagation and
  snare ring-resonance dynamic/static notch behavior with measured ring
  parameter propagation,
  solo bass low-end resonance dynamic/static selection plus
  declared-vocal/low-end-audio identity conflict hold behavior.
- Replay evaluation tests for pass rate, failure reporting, artifact writing
  and CLI behavior, including non-empty `alternative_tradeoffs` and required
  `operator_explanation_sections` plus method-specific future WING readback
  sections when a scenario has competing candidates.
- Proposal contract tests to keep operator-review exports valid.
- Capability matrix tests to confirm the method is not a live executor.
- Full backend suite before treating the checkpoint as verified.

## Risks And Rollback

Risk: the initial scoring is heuristic and not yet calibrated against replay
quality thresholds.

Rollback: remove the new engine, registry entry, proposal contract entry and
focused tests. This leaves existing per-method proposal logic unchanged.

Safety: this ADR grants no WING authority, no queue import, no live apply and no
readiness closure.
