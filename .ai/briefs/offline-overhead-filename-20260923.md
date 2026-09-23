# Offline overhead identity and balance diagnostic

## Thesis and observed problem

The canonical eight-track render classifies `04_overhead_L_066_phrase_rock_complex_fast_sticks.wav`
and its R counterpart as `custom`, both at pan 0. The existing overhead-pair
spatial and phase analysis is consequently skipped. This is source metadata
parsing failure, not a request to invent new DSP or infer quality from tests.

## Proposed change

Recognize bounded overhead/OH side tokens with normal filename separators in
the existing offline classifier; use the same interpretation when choosing the
overhead pair. Keep the existing overhead profiles and signal-based panning.
Do not normalize every instrument name or change the live classifier.

## Files and alternatives

`tools/offline_agent_mix.py`, its existing classification tests, report and ADR.
Renaming user audio is rejected: it hides the defect. General classifier
replacement is outside this measured failure. No new dependencies.

## Risk and validation

False side matches can swap the image. Test bounded tokens, separator variants,
legacy OH names, explicit sides overriding incidental pan ordering, and unknown
names. Verify a real drum render and repeat the unchanged 16-track control.
Compare audio metrics without claiming subjective improvement.

## Independent disagreement

Whether filename metadata is sufficiently trustworthy remains an operator
question. This patch only honors explicit existing names in an offline runner.
Kimi review was previously unavailable (403 subscription); independent review
is not claimed. Shared-file patch is unilateral pending Claude review.

## Parallel diagnostic

On source HEAD 79ed951f3, run control and one-source -6 dB trials on both real
datasets. Renderer receives no clean target or perturbation metadata. Disabling
SmartDSP sidecar affects reporting only; verify control WAV hashes against the
previous complete pipeline. Raw experiment lives under
`/tmp/automixer-balance-trial-20260923.qwzeNh`.
