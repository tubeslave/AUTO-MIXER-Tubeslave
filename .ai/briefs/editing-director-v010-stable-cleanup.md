# Editing Director v0.10: Stable DeNoise / DeBleed

Reason for redesign: v0.9 gate envelopes were audibly floating. The cleanup stage must not pump or chatter.

New strategy:
- no real-time gate state and no continuously chasing threshold;
- vocals: phrase-region detection, merged regions, long crossfades, fixed -14 dB attenuation outside confident phrases;
- kick/snare/toms: confirmed event windows with fixed attenuation between events; longer protected decays; no threshold flutter;
- bass/guitar: note/body-region clip gain plus stationary spectral denoise using one fixed noise profile for the whole song;
- hi-hat: high-frequency activity regions with fixed attenuation outside them;
- overheads stay natural;
- Keys/Playback are unchanged unless their own noise analysis requires cleanup.

The region editor is deterministic clip-gain editing, not a live gate. The denoiser uses a fixed learned noise profile, so its character does not change section by section.

QA goals:
- contamination/quiet-region reduction about 6–12 dB where justified;
- strong musical material normally changes <1 dB;
- preserve drum attack/decay windows;
- no new clicks at region boundaries;
- human A/B is mandatory before the cleaned multitrack becomes the new pipeline baseline.

## 2026-09-23: full-song model cleanup validation checkpoint

The first sequential full-song cleanup was not acceptable as infrastructure: it ran all model jobs serially on CPU, hit the 180-minute GitHub Actions timeout and lost the unpublished work. The workflow is now checkpointed per source.

Replacement run `35846566268` completed successfully:
- 11/11 model-cleaned sources finished and uploaded independent checkpoints;
- every source checkpoint contains the cleaned WAV, full-length RAW/CLEAN/REMOVED MP3 previews and residual JSON;
- the final checkpointed render also completed and uploaded artifact `ptitsa-full-model-cleanup-mix` (`10746216442`), containing `Ptitsa_ModelClean_FullMix_50pct.wav` plus a 320 kbps listening MP3 and the validation previews.

Objective residual metrics are only a screening signal, not an acceptance decision. They show a clear intervention split:
- `NIKITA_VOX`: removed/raw RMS `-6.62 dB`;
- `VALERA_VOX`: removed/raw RMS `-7.62 dB`;
- non-vocal model-cleaned sources: roughly `-29.38 dB` to `-52.49 dB` removed/raw.

Therefore both vocal cleanups are explicitly high-risk for over-cleaning and must be judged from RAW/CLEAN/REMOVED listening before the processing is accepted, weakened or rejected. No model-cleaned source becomes the baseline merely because the workflow succeeded.

## Editing regression guards discovered during CI

Two studio regressions were exposed while the long cleanup render was running:

1. `click_candidates` missed a one-sample impulse because the impulse creates two adjacent derivative spikes and each spike was being treated as local evidence against the other. Commit `569c856350b7f8f7e0368e6a30a57b13851f3716` clusters adjacent supra-threshold derivatives, compares the cluster against samples outside it and rejects broad transient runs. Regression fixtures cover an impulse, a step discontinuity, a broad ramp and a smooth periodic signal. Full Tests run `35852601258` progressed past this test and reached the later guitar-director test after 344 passes and one skip on Python 3.11.

2. `wall_layer` used a fixed 9 kHz upper band edge, which is invalid in the reduced-rate 1 kHz fixture. Commit `7e4d395b9631abf14bfc8bda3c2b98a90ef8a6fb` caps the wall-layer band below Nyquist and fails closed to silence if the intended 120 Hz lower edge cannot be represented. The fixture was also corrected from DC to an in-band guitar-like sine, so it actually exercises dense-section wall energy rather than a signal the bandpass should remove.

Current policy remains fail-closed: objective checks can reject or flag a cleanup, but subjective source quality still requires human listening.
