# Cleanup Candidates For Decision Layer

No files were removed. These are only candidates for future review:

- `tools/offline_agent_mix.py`: very large legacy offline mixer with many
  responsibilities in one file. It contains useful DSP and candidate-search
  ideas, but should be wrapped or mined into focused modules before any cleanup.
- `external/automix-toolkit`: appears as untracked/dirty local reference code.
  Keep it as research input until ownership and provenance are clear.
- Previous generated run folders under `/Users/dmitrijvolkov/Desktop/Ai MIXING`:
  they are artifacts, not source. Do not delete without operator request.
- Old experimental MuQ-A1 candidate banks: useful as regression material for
  critic/decision replay, but not production code.
