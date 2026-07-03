# Snare Bleed / Global Phase Alignment

`snare_bleed_phase_alignment` is the first offline phase/delay method used by
the `ayaic-clean` render pipeline when a usable snare reference exists.
`global_bleed_phase_alignment` remains as fallback for material without reliable
snare evidence.

## Goal

The primary method aligns microphones by the arrival of the snare drum:

- detect snare transients from `SNARE T`/`SNARE TOP` when available;
- measure the snare bleed arrival in every reliable microphone channel;
- keep the latest reliable snare arrival at `0 ms`;
- delay earlier arrivals with positive offline fractional delay;
- check polarity only after the delay pass.

The fallback global method aligns channels that share strong acoustic
information:

- drum close mics against overhead/room bleed;
- stereo microphone pairs;
- vocal, guitar, keys, accordion, playback, and backing vocal channels when
  they carry the same leakage or highly correlated spill;
- other cross-bleed pairs that pass confidence gates.

Neither method blindly aligns every channel. Low-correlation or unstable
measurements are reported as rejected and are not applied.

## Method

### Snare-first method

1. Select snare reference:
   - prefer `SNARE T` / `SNARE TOP`;
   - otherwise use the best available snare mic.
2. Detect short snare transient windows.
3. For each candidate microphone:
   - band-limit to snare-relevant content;
   - measure event-wise delay against the snare reference with GCC-PHAT;
   - compute confidence, coherence, PSR, and stability across hits.
4. Reject weak/non-microphone evidence:
   - no valid snare windows;
   - unstable arrival across hits;
   - weak GCC peak;
   - low confidence/correlation.
5. Choose the reliable channel with the latest snare arrival as `zero_ms_source`.
6. Apply:

```text
delay[channel] = latest_arrival_ms - arrival_ms[channel]
```

7. Clamp by `max_phase_delay_ms`.
8. Apply fractional delay offline.
9. Check polarity against the zero-ms source.
10. Validate before/after snare alignment score and roll back if it gets worse.

### Global fallback method

1. Build a correlation graph across all stems.
2. Measure candidate pair delays with multi-window GCC-PHAT.
3. Reject weak or unstable edges using:
   - correlation strength;
   - delay stability across windows;
   - GCC peak-to-second-peak ratio;
   - confidence score;
   - relation type, such as stereo pair, same group, ambience reference, or
     cross-bleed.
   - drum close mic direction: close mic -> overhead/room must indicate the
     close mic is earlier; the method will not delay overheads just to satisfy a
     weak or contradictory close-mic edge.
4. Solve each connected component as weighted least squares:

```text
delay[target] - delay[source] = measured_relative_offset
```

5. Convert relative offsets into non-negative offline delays by shifting each
   component so its earliest channel is `0 ms`.
6. Apply fractional delay in offline audio.
7. Run a polarity pass against the weighted correlation graph.
8. Validate before/after graph correlation and roll back if the score gets worse.

## Safety

- The method is offline-only.
- It does not send OSC commands.
- `max_phase_delay_ms` still caps the component spread.
- Snare bleed candidates with low confidence are rejected.
- The latest reliable snare arrival stays at `0 ms`; the method never requires
  negative delay.
- Channels without reliable snare leakage stay unchanged.
- Fallback global graph edges with low confidence are rejected.
- Room/overhead and generic cross-bleed edges use stricter gates/lower weights
  to avoid collapsing natural space.
- Drum close mics can be delayed against overhead/room references, but
  overhead/room channels are not moved by a single negative close-mic edge.
- Rollback restores original audio, delay, and polarity if global correlation
  validation fails.

## Report Fields

The pipeline report includes:

- `phase_delay_order`
- `overhead_pair_alignment`
- `phase_alignment`
- `snare_bleed_alignment`
- `global_phase_alignment`

`snare_bleed_alignment` includes the snare reference, transient count,
zero-ms source, candidate/rejected microphone lists, latest arrival, before/after
alignment score, and rollback status.

`global_phase_alignment` includes candidate edge count, rejected edges, connected
components, solved offsets, before/after graph score, rollback status, and the
edge list used for the solve.

## Evidence Base

The delay estimator follows the GCC/GCC-PHAT family of methods. The broad
multi-microphone approach mirrors professional auto-alignment tools that use
time offset and polarity correction across grouped tracks. Audible improvement
from automatic phase alignment is program-dependent; for a universal guarantee
there is no strong evidence.

References:

- C. H. Knapp and G. C. Carter, "The Generalized Correlation Method for
  Estimation of Time Delay", IEEE Transactions on Acoustics, Speech, and Signal
  Processing, 1976.
- MathWorks `gccphat` documentation for generalized cross-correlation / TDOA.
- Sound Radix Auto-Align 2 documentation and manual for real-world grouped
  microphone time/phase alignment workflows.
