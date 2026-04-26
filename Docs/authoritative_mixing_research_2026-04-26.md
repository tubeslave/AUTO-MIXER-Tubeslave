# Authoritative Mixing Research Notes - 2026-04-26

## Scope

Sources reviewed for this pass:

- JAES/AES research on masking reduction, panning, and loudness-based automatic
  level setting.
- arXiv research on differentiable/controllable automatic mixing.
- EBU/ITU official loudness and true-peak standards.
- Sound On Sound professional articles by Mike Senior and Paul White.
- Vetted engineer/manufacturer YouTube ecosystems only as source registries or
  when a reviewed official article page was available.

## Notes

1. Masking should be reduced as a specific problem, not erased globally. JAES
   work by Hafezi/Reiss reports masking-reduction EQ can improve perceived
   quality, while AES work from iZotope frames problematic masking as
   instrument/frequency outliers rather than all overlap.

2. Panning is a masking tool. AES spatialization work treats stereo placement as
   a way to improve clarity, but anchors still matter: kick, bass, snare, lead
   vocal, and sub-low energy should remain stable and mono-compatible.

3. Loudness features are useful for initial automatic level balance. AES
   research on automatic level adjustment supports loudness/energy descriptors,
   but EBU/ITU standards define measurement and safety descriptors rather than
   artistic balance rules.

4. Compression should follow a musical reason. SOS guidance repeatedly warns
   that compression can raise room tone and noise; volume automation or smaller
   gain moves are often more transparent for large vocal swings.

5. Drums need density without flattening. SOS drum material supports blended or
   parallel density paths when the kit needs weight, while preserving the main
   transient image.

6. Compressor timing is goal-dependent. Slower kick attack can keep click while
   compressing body; very fast timing can add snare/body density but risks
   pulling bleed and cymbals forward.

7. Depth comes from level, tone, delay, reverb, and pre-delay together. In busy
   rock mixes, filtered sends and delay/predelay often create vocal space more
   safely than simply raising a wet reverb.

8. Low end starts with headroom and role separation. True peak safety must be
   checked before low boosts; kick weight and bass sustain need distinct roles.

9. Vocal edge does not have to mean large high-frequency EQ. Producer guidance
   from Produce Like A Pro points toward saturation/harmonic density as an
   alternative path when esses and consonants become unstable.

10. Learned or MERT-like systems should stay inspectable. arXiv automatic mixing
    work argues for traditional, human-readable control parameters, supporting
    the project choice to log source IDs, before/after metrics, and actions.

11. Drum phase/time-alignment should be an auditioned candidate, not an
    automatic visual correction. AES and SOS sources agree that phase problems
    can masquerade as EQ/compression problems, but small delay or polarity
    changes must be judged against punch, body, mono compatibility, bleed, and
    the full mix.

12. Reverb and delay returns should behave like controlled shared spaces.
    SOS/FabFilter guidance favors fully wet sends, return filtering, and full
    mix level checks; delay can create depth with less masking than long reverb
    in dense arrangements.

13. Pre-delay is a role-specific clarity tool. Manufacturer education suggests
    using it to let dry attacks speak before the ambience blooms, with vocals,
    guitars, and drums needing different audition ranges rather than one fixed
    value.

14. Modulation effects need role limits. Chorus is useful for support width,
    phaser for more nuanced movement, and flanger for stronger comb-filter
    motion or momentary effects. Placing modulation before reverb textures the
    tail; after reverb it moves the whole ambience.

15. Early reflections and late reverb should be shaped separately. Lexicon and
    Valhalla documentation connect early/pre-echo behavior to perceived
    reflecting surfaces, while diffusion/spin/wander affect density, naturalness,
    and motion without intending to destabilize source position.

## Project Encoding

Added source metadata and paraphrased rules to:

- `backend/source_knowledge/data/sources.yaml`
- `backend/source_knowledge/data/rules.jsonl`

New rule IDs:

- `masking.problem_outliers_only`
- `panning.unmask_supports_keep_anchors`
- `levels.simple_loudness_then_context`
- `dynamics.preserve_microdynamics_first`
- `drums.parallel_weight_without_flattening`
- `drums.attack_body_compressor_timing`
- `fx.depth_via_filtered_delay_predelay`
- `low_end.gainstage_weight_click_headroom`
- `vocal.edge_saturation_before_hf_overboost`
- `automation.keep_effect_parameters_interpretable`
- `phase.drum_time_alignment_audition`
- `fx.shared_filtered_returns_context`
- `fx.predelay_by_role_preserve_attack`
- `fx.delay_depth_when_reverb_masks`
- `fx.ducked_returns_front_clarity`
- `fx.modulation_support_width_texture`
- `fx.modulation_reverb_order_intent`
- `fx.early_late_density_by_role`

## Sources

- Hafezi & Reiss, JAES, "Autonomous Multitrack Equalization Based on Masking
  Reduction": https://doi.org/10.17743/jaes.2015.0021
- AES 146, Tom/Reiss/Depalle, spatialization/unmasking:
  https://secure.aes.org/forum/pubs/conventions/?elib=20311
- AES 141, Wichern/Robertson/Wishnick, loudness-loss masking:
  https://secure.aes.org/forum/pubs/conventions/?elib=18450
- AES 139, Wichern/Wishnick/Lukin/Robertson, loudness features:
  https://secure.aes.org/forum/pubs/conventions/?elib=17928
- Steinmetz et al., arXiv 2010.10291:
  https://arxiv.org/abs/2010.10291
- Vanka et al., arXiv 2407.08889:
  https://arxiv.org/abs/2407.08889
- EBU R 128:
  https://tech.ebu.ch/publications/r128
- ITU-R BS.1770:
  https://www.itu.int/rec/R-REC-BS.1770/
- Sound On Sound, "Mixing Essentials":
  https://www.soundonsound.com/techniques/mixing-essentials
- Sound On Sound, "Creating A Sense Of Depth In Your Mix":
  https://www.soundonsound.com/techniques/creating-sense-depth-your-mix
- Sound On Sound, "Mixing Multitracked Drums":
  https://www.soundonsound.com/techniques/mixing-multitracked-drums
- Sound On Sound, "Mixing Metal":
  https://www.soundonsound.com/techniques/mixing-metal
- Pensado's Place, "Into The Lair":
  https://www.pensadosplace.tv/intothelair/
- Produce Like A Pro, quick mixing tricks archive:
  https://producelikeapro.com/blog/tag/quick-mixing-tricks/
- AES 140, Kruk/Sobecki, phase-aligning multi-mic recordings:
  https://secure.aes.org/forum/pubs/conventions/?elib=18278
- Sound On Sound, "Making Multi-mic Recordings Work":
  https://www.soundonsound.com/techniques/making-multi-mic-recordings-work
- Sound On Sound, "Mix Rescue: Phase Relationships":
  https://www.soundonsound.com/techniques/mix-rescue-phase-relationships
- Sound On Sound, "Using Reverb & Delay":
  https://www.soundonsound.com/techniques/using-reverb-delay
- Sound On Sound, "How To Use Reverb Like A Pro: 1":
  https://www.soundonsound.com/techniques/how-use-reverb-pro-1
- Sound On Sound, "How To Optimise Your Reverb Treatments":
  https://www.soundonsound.com/techniques/how-optimise-your-reverb-treatments
- Sound On Sound, "How Phasers Work":
  https://www.soundonsound.com/sound-advice/how-phasers-work
- Sound On Sound, "Q. What's The Difference Between Phasing And Flanging?":
  https://www.soundonsound.com/sound-advice/q-whats-difference-between-phasing-and-flanging
- iZotope, "Understanding Chorus, Flangers, and Phasers in Audio Production":
  https://www.izotope.com/en/learn/understanding-chorus-flangers-and-phasers-in-audio-production.html
- iZotope, "Reverb Pre-delay Explained":
  https://www.izotope.com/en/learn/reverb-pre-delay.html
- FabFilter, "Reverb - How To Use Reverb?":
  https://www.fabfilter.com/learn/reverb/how-to-use-reverb
- Valhalla DSP, "ValhallaRoom: The Early Controls":
  https://valhalladsp.com/2011/05/18/valhallaroom-the-early-controls/
- Lexicon 480L Digital Reverb and Effects Manual:
  https://help.uaudio.com/hc/en-us/articles/33194625601044-Lexicon-480L-Digital-Reverb-and-Effects-Manual
- Eventide H90 Documentation, "Modulation":
  https://cdn.eventideaudio.com/manuals/h90/1.9.4/content/algorithms/modulation.html
