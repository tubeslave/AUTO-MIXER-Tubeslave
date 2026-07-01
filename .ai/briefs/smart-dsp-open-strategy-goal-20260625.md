# Brief: Smart DSP Open Strategy Goal

## Goal

Build Automixer's smart DSP decision engine so it can find the fastest, most
accurate and safest processing strategy for soundcheck and live-assist work,
without being locked to the DSP methods that already exist in the repository.

Existing EQ, compressor, gate, FX, pan, gain, group, master/output or other DSP
methods are not mandatory architecture. If evidence shows that another approach
is faster, more precise, safer, easier for the operator to verify, or easier to
roll back, the engine may replace the existing method, rework it, or add a
meta-selector that chooses a better strategy.

The current hardware target is Behringer WING Rack firmware 3.0.5. Any
correction that can be promoted toward WING application must exist as a native
WING Rack fw3.0.5 console correction and must have an explicit OSC value/path
and readback contract. Smarter analysis may rank or explain non-native ideas,
but those ideas remain review-only/dry-run unless they reduce to native WING
fw3.0.5 OSC corrections.

## Operating Principle

The engine should solve the audible and operational problem, not defend a fixed
method list.

For every detected issue, it should compare practical alternatives and score
them on:

- speed of useful correction;
- precision against the measured problem;
- musical usefulness for the source role;
- live safety and headroom impact;
- reversibility and rollback clarity;
- operator explanation quality;
- readiness for future WING readback/proof.

The winner is the best safe action, not necessarily the most complex DSP action.
If the safest correct answer is level-plane correction, operator review,
readback recheck, source confirmation, or doing nothing, that must beat an EQ,
compressor, gate or FX change.

## Soundcheck Scope

Soundcheck can be exploratory and evidence-led. It may compare offline/dry-run
strategies across channel, group and master/output problems.

Allowed soundcheck outputs are bounded proposals only:

- level-plane and gain-staging corrections;
- EQ cuts or carefully bounded boosts;
- dynamic EQ, de-essing, compression, expansion and gating;
- FX send/return, decay/filter, delay feedback and ducking proposals;
- pan/width proposals with phase and mono checks;
- group and master/output trim, EQ, dynamics or limiter proposals when
  headroom and true-peak evidence support them.

Soundcheck success means finding a fast, explainable and safe starting point for
a musical mix. It does not mean the system is ready to write to the real mixer.
Soundcheck apply candidates must still be WING Rack fw3.0.5 native OSC only;
non-native, plugin-dependent, unverified sidechain, or unmapped FX/model
parameters are explanation/review artifacts, not apply-ready corrections.

## Live Scope

Live mode must be more conservative than soundcheck.

The preferred live action is usually risk reduction:

- gain-down rather than gain-up;
- narrow cuts rather than broad tonal rewrites;
- gentle, bounded dynamics rather than aggressive compression/gating;
- FX reduction rather than detailed time/decay edits;
- hold/review when evidence, source identity, readback or rollback is weak.

Without real WING proof, live EQ, compressor, gate, FX, pan, routing, scene,
snapshot and master/output writes remain forbidden. The engine may form
operator-review proposals and future readback contracts, but it must not widen
live authority.
Even with future proof, live promotion must stay inside corrections exposed by
WING Rack fw3.0.5 OSC for the target surface; unsupported or unverified
processing remains blocked regardless of score.

## Required DSP Reasoning

Each method family must be treated as a candidate strategy, not as a default
answer:

- vocal mud: compare HPF, low-mid cut, dynamic EQ, multiband, level-plane
  correction and no-EQ hold;
- vocal harshness or sibilance: compare de-esser, dynamic EQ, presence cut,
  compressor envelope adjustment and split-chain control;
- kick/bass conflict: compare pocket EQ split, keyed dynamic ducking,
  level-plane rebalance and arrangement-aware hold;
- noisy toms/drums: compare expander, gate, mute review, bleed-aware threshold
  and live hold behavior;
- FX intelligibility loss: compare send/return reduction, decay/filter cleanup,
  return ducking, delay feedback reduction and feedback-risk blocking;
- pan/width problems: compare pan correction, width narrowing, mono-safety hold
  and operator review;
- master/output overload: compare trim, group rebalance, limiter/compressor
  review and broad EQ only after true-peak/headroom checks.

## Evidence Required

The engine must base decisions on current evidence where possible:

- source identity and musical role;
- level, LUFS, peak and true peak;
- crest factor and LRA proxy;
- spectral balance and resonance peaks;
- masking and source-pair conflicts;
- transient behavior and envelope timing;
- noise floor, bleed and gate tail/chatter risk;
- phase and mono compatibility;
- feedback risk;
- FX wetness, tail and delay smear;
- group, bus and master/output impact.

Low-confidence, stale or conflicting evidence must select a hold/recheck
strategy before normal DSP.

## Parameter Requirements

Selected candidates must calculate concrete, bounded parameters:

- EQ/dynamic EQ frequency, gain, Q, filter type, threshold and max reduction;
- compressor threshold, ratio, attack, release, knee, makeup and target gain
  reduction;
- gate/expander threshold, range, attack, hold, release and hysteresis;
- de-esser band, Q, threshold, range, attack and release;
- FX send/return delta, decay/filter/delay/ducking parameters;
- pan/width target and delta;
- trim/fader/group/master bounded delta.

Fallback defaults are allowed only when the missing measurement is explicit in
the explanation and safety score.

## Operator Contract

Every proposal must explain:

- what problem was detected;
- why this processing type was selected;
- which alternatives lost and why;
- what exact parameters are proposed;
- what risks remain;
- what the operator must verify;
- what readback is expected;
- how rollback works.

## Hard Boundaries

Do not:

- treat offline/dry-run success as concert readiness;
- promote a correction to WING-applyable unless it exists on Behringer WING
  Rack fw3.0.5 and is represented by explicit native OSC value/path/readback
  fields;
- expand WING live authority without real hardware proof;
- perform live EQ, compressor, gate, FX, pan, routing, scene, snapshot or
  master/output writes;
- bypass source identity, stale-evidence, feedback, headroom, readback or
  rollback guards;
- call a method complete if it only creates a card but does not make a tested
  smart DSP decision.

## Completion Criterion

This goal is complete only when Automixer has a proven offline/dry-run smart DSP
engine that can compare alternative processing strategies, choose the best safe
one for soundcheck and live-assist scenarios, calculate bounded parameters,
explain the decision, block unsafe cases, preserve the current WING safety
boundary, restrict apply-ready corrections to native Behringer WING Rack fw3.0.5
OSC capabilities, and pass focused plus full tests.
