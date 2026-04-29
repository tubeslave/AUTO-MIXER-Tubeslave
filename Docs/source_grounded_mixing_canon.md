# Source-Grounded Mixing Canon

This note summarizes the source groups that should guide offline and live-safe
automatic mixing decisions. It stores paraphrased principles only: no long book,
article, interview, or seminar excerpts are kept in the repository.

## Practical Mixing Books

- Balance first, processing second. Faders, pan, mute/edit choices, and source
  roles should establish the musical picture before corrective EQ, compression,
  effects, or master-bus processing.
- Every processor needs a reason. EQ solves tone, masking, translation, or
  feedback risk; compression solves envelope, level variation, density, or
  sustain; FX solves depth, width, texture, motion, or special color.
- The mix is a sound stage. Important anchors such as lead vocal, kick, snare,
  and bass normally remain stable; support elements create width and depth
  around them.
- Depth is multi-factor: level, tone, dry/wet balance, predelay, early/late
  reverb balance, delay, and automation all contribute.
- Avoid formula worship. Use book and interview guidance as listening prompts
  and bounded starting points, not as fixed recipes.

Primary source IDs:
`izhaki_mixing_audio`, `senior_mixing_secrets`, `owsinski_handbook`,
`moylan_crafting_mix`, `gibson_art_of_mixing`,
`stavrou_mixing_with_your_mind`, `huber_modern_recording`,
`rumsey_sound_recording`.

## Mastering, Monitoring, And Translation

- Mastering should preserve intent, headroom, and dynamics. Louder is not
  automatically better.
- LUFS and true peak are technical descriptors. They support safety and
  comparison, but they do not define musical quality by themselves.
- Prefer static trim and conservative bus treatment before limiting. Reject
  clipping, obvious distortion, aggressive pumping, and abrupt tonal changes
  even if a metric score improves.
- Playback systems and rooms bias judgment. Use mono checks, true-peak checks,
  crest factor, spectral balance, and operator feedback before trusting a single
  monitor path.

Primary source IDs:
`katz_mastering_audio`, `vickers_loudness_war`,
`deruty_tardieu_dynamic_processing`, `itu_bs1770`, `ebu_r128`,
`toole_sound_reproduction`, `yamaha_sound_reinforcement`.

## Intelligent Music Production And Automatic Mixing

- Automatic systems must remain bounded, inspectable, and reversible.
- Decisions should be represented as conventional console parameters: gain,
  fader, pan, EQ, dynamics, FX send/return, bus, or master parameters.
- Log before/after metrics, source rule IDs, safety state, accepted/rejected
  actions, and operator feedback.
- Objective metrics and learned evaluators are feedback layers, not direct
  mixer controllers.
- Offline experiments must prevent reference leakage: hidden final mixes,
  filenames, dataset labels, and undeclared references cannot be decision
  shortcuts.

Primary source IDs:
`intelligent_music_production`, `moffat_sandler_approaches_imp`,
`reiss_intelligent_systems_multichannel`,
`de_man_reiss_knowledge_engineered_mixing`,
`perez_reiss_live_fader_control`, `mansbridge_pan_positioning`,
`safe_semantic_audio_features`, `arxiv_steinmetz_differentiable_mixing`,
`arxiv_diff_mst`, `medleydb_dataset`, `musdb18_dataset`.

## Metrics, Psychoacoustics, And Quality Evaluation

- Use perceptual metrics only inside their domain assumptions and calibrate
  thresholds against references and listening feedback.
- MUSHRA/BS.1116 style thinking matters: human listening remains the ground
  truth for small quality differences.
- Masking and timbre features identify candidate problems; they should not make
  broad tonal moves without persistence across sections and full-mix context.
- SDR/source-separation metrics are not mix-quality metrics. They can help
  evaluate isolation or artifacts, but not artistic balance alone.

Primary source IDs:
`itu_bs1534_mushra`, `itu_bs1116`, `itu_bs1387_peaq`, `thiede_peaq`,
`hines_visqol_audio`, `peeters_timbre_toolbox`, `mcfee_librosa`,
`vincent_bss_eval`, `le_roux_sdr_half_baked`,
`moore_psychology_hearing`, `zwicker_fastl_psychoacoustics`.

## Professional Corpora

- Use Sound On Sound, Tape Op, Mix With The Masters, Pensado material, and
  engineer interviews for general transferable principles: intent, contrast,
  arrangement awareness, A/B discipline, automation, and context listening.
- Do not imitate a named engineer, artist, or trademark sound. Translate any
  style direction into measurable generic attributes, or use an operator-provided
  reference track.

Primary source IDs:
`sos_mix_rescue_series`, `sos_secrets_mix_engineers`, `sos_inside_track`,
`tape_op_interviews`, `mix_with_the_masters_platform`, `pensado_papers`,
`massey_behind_the_glass`.

## Current Rule Hooks

The distilled principles above are encoded in
`backend/source_knowledge/data/rules.jsonl`, especially:

- `workflow.balance_first_processing_second`
- `balance.level_planes_role_context`
- `soundstage.depth_width_frequency_picture`
- `mastering.transparent_headroom_over_loudness`
- `monitoring.translation_room_suspicion`
- `psychoacoustics.masking_loudness_context`
- `automation.human_metric_hybrid_decisions`
- `automation.conventional_controls_interpretable`
- `datasets.no_reference_or_filename_leakage`
- `effects.intent_before_preset_name`
- `professional_principles_no_style_imitation`
- `arrangement.mix_moves_should_respect_source_role`

These rules are advisory or shadow by design. They may influence offline renders
and logged candidate choices, but live OSC changes must still pass the existing
safety, rate-limit, headroom, and operator-review layers.
