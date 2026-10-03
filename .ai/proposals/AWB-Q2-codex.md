# AWB-Q2 proposal

## Thesis
Quality will improve more from context and causal testing than from adding more global metrics.

## Solution
Add a schema-backed hierarchy, event descriptors and a causal experiment plan that wraps the existing reversible renderer. Keep measurement, hypothesis and preference separate.

## Files
audio_workbench/song_model.py
audio_workbench/events.py
audio_workbench/causal.py
audio_workbench/server.py
tests/test_audio_workbench_song_model.py
tests/test_audio_workbench_events.py
tests/test_audio_workbench_causal.py

## Risks
Filename-derived grouping can be wrong, so guesses never become protected musical intent without explicit confirmation/context. Generic onset detection is not source classification. Causal probes are bounded and cannot prove artistic preference.

## Test plan
Synthetic stems and impulses; no external model weights required.
