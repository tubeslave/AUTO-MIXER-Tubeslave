---
dataset_info:
  features:
  - name: mix_name
    dtype: string
  - name: song_name
    dtype: string
  - name: artist_name
    dtype: string
  - name: genre
    dtype: string
  - name: track_name
    dtype: string
  - name: track_type
    dtype: string
  - name: track_instrument_subtype
    dtype: string
  - name: track_instrument_type
    dtype: string
  - name: channel_mode
    dtype: string
  - name: track_audio_path
    dtype: string
  - name: track_audio_sample_rate
    dtype: int64
  - name: track_audio_lufs
    dtype: float64
  - name: parameters
    dtype: string
  splits:
  - name: train
    num_bytes: 771806.2324093817
    num_examples: 1545
  - name: dev
    num_bytes: 149865.28784648186
    num_examples: 300
  - name: test
    num_bytes: 249775.47974413645
    num_examples: 500
  download_size: 284870
  dataset_size: 1171447.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: dev
    path: data/dev-*
  - split: test
    path: data/test-*
---
