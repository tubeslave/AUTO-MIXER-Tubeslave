# Ptitsa model-cleanup benchmark

Automatically selected difficult 24-second excerpts from the original synchronized multitrack:

- Valera vocal: 179–203 s. Purpose: vocal de-noise/de-bleed.
  References: OH L/R, guitar, playback L/R.
- Snare top: 35–59 s. Purpose: cymbal/kit bleed removal while preserving snare attack/tail.
  References: snare bottom, OH L/R, hi-hat.
- Bass: 13–37 s. Purpose: stationary noise + inter-note scrape/noise.
  References: kick in/out, guitar.
- Guitar: 13–37 s. Purpose: amp/string noise + inter-note scrape.
  References: vocals, OH L/R.

Acceptance requires RAW/CLEAN/REMOVED audition. No full-song processing until one candidate passes on these excerpts.

IRMR inspection:
- public tUNet folder contains ArtificialMix.py, CM_Generate.py, CM_GenerateTest.py, getSDR.py and train.py;
- no pretrained .h5/checkpoint is present;
- README expects the user to train and then pass --model /path/to/model.

Therefore IRMR is architecture/reference code, not a ready pretrained de-bleed processor.

Current external-model priority:
1. released music-debleed model with weights, if found;
2. DeepFilterNet only for vocal stationary-noise benchmark, never as de-bleed;
3. song-specific weak/self-supervised training only as a separate experiment.
