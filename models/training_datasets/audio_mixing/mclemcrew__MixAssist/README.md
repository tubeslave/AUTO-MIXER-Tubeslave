---
dataset_info:
  features:
  - name: conversation_id
    dtype: string
  - name: topic
    dtype: string
  - name: turn_id
    dtype: int64
  - name: has_content
    dtype: bool
  - name: input_history
    list:
    - name: audio_file
      dtype: string
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: audio_file
    dtype: string
  - name: user
    dtype: string
  - name: assistant
    dtype: string
  splits:
  - name: train
    num_bytes: 1112009
    num_examples: 340
  - name: test
    num_bytes: 1129399
    num_examples: 250
  - name: validation
    num_bytes: 59468
    num_examples: 50
  download_size: 192053
  dataset_size: 2300876
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
---
