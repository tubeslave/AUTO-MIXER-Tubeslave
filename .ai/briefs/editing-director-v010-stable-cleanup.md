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
