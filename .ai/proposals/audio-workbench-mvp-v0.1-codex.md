# Audio Workbench MVP hardening proposal

## Thesis
Make Workbench fail closed and reuse Automixer safety rather than building another control stack.

## Solution
Harden render identity/check evidence, strict A/B shape validation, float/non-destructive rendering, calibration-enforced plugin hosting, minimal aligned multitrack summing, runtime doctor, and a non-applying bridge into MixAgentBackendBridge.

## Risks
The renderer is intentionally small: aligned mono/stereo files, static gain/pan/polarity. This is safer than pretending to be a full DAW. Stereo pan is rejected when ambiguous.

## Test plan
Synthetic regression tests for every previously reproduced defect plus repository CI.
