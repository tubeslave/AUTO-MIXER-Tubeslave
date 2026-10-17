# STUDIO brief — Bass Stability Gate v1

## Why now
The user accepted the Belye Stai v2 vocal compression for stable foreground placement and asked for instrument-specific compression. The first generic bass family reduced first-stage GR and worsened measured body-level consistency. A second stronger-GR family also worsened the same fixed-window proxy. We need a bass-specific fail-closed gate before changing the accepted mix.

## Bounded task
Add a baseline-aware bass compression director that:
- uses the existing bass first compressor as an explicit no-change reference;
- measures candidate body spread and attack/body contrast on fixed macro-event windows derived only from the pre-compression source;
- proposes small attack/release-only probes while freezing threshold, ratio, knee, detector, RMS integration and max-GR;
- can reject every probe and keep no-change;
- never ranks a musical winner or promotes a baseline without human listening.

## Out of scope
No live/OSC work, no mastering changes, no neural audio, no paid external services, no change to the accepted Belye Stai vocal, and no automatic release promotion.
