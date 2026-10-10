# Mixing Director v1.4: Drum Director

Accepted baseline: v1.3 Space.

This stage improves drum authority without changing the accepted balance architecture.

- detect main kick/snare hits from the already edited/dynamics-controlled close mics;
- build reinforcement samples from the recording itself by median-stacking the cleanest/strongest hits;
- no external sample library and no new drum timbre is invented;
- reinforce only confirmed main hits;
- kick reinforcement is stronger than snare by default;
- keep reinforcement low enough to add density/punch rather than replace the drummer;
- preserve overheads and room perspective;
- re-render the shared Space Director after drum changes;
- Drum Critic checks attack gain, drum-bus RMS gain and global spectral shift;
- level-matched A/B against v1.3 is mandatory.

This is the first Instrument Director. Vocal, Bass and Guitar Directors follow separately so each layer can be accepted or rolled back.
