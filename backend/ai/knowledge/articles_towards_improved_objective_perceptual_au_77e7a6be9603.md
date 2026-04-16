<!-- source_type: articles -->
# Towards Improved Objective Perceptual Audio Quality Assessment -- Part 1: A Novel Data-Driven Cognitive Model

- source_type: articles
- source: arxiv
- url: http://arxiv.org/abs/2411.18222v1
- discovered_at: 2026-04-16T22:27:35.818231+00:00

## Notes

Efficient audio quality assessment is vital for streamlining audio codec development. Objective assessment tools have been developed over time to algorithmically predict quality ratings from subjective assessments, the gold standard for quality judgment. Many of these tools use perceptual auditory models to extract audio features that are mapped to a basic audio quality score prediction using machine learning algorithms and subjective scores as training data. However, existing tools struggle with generalization in quality prediction, especially when faced with unknown signal and distortion types. This is particularly evident in the presence of signals coded using non-waveform-preserving parametric techniques. Addressing these challenges, this two-part work proposes extensions to the Perceptual Evaluation of Audio Quality (PEAQ - ITU-R BS.1387-1) recommendation. Part 1 focuses on increasing generalization, while Part 2 targets accurate spatial audio quality measurement in audio coding. To enhance prediction generalization, this paper (Part 1) introduces a novel machine learning approach that uses subjective data to model cognitive aspects of audio quality perception. The proposed method models the perceived severity of audible distortions by adaptively weighting different distortion metrics. The weights are determined using an interaction cost function that captures relationships between distortion salience and cognitive effects. Compared to other machine learning methods and established tools, the proposed architecture achieves higher prediction accuracy on large databases of previously unseen subjective quality scores. The perceptually-motivated model offers a more manageable alternative to general-purpose machine learning algorithms, allowing potential extensions and improvements to multi-dimensional quality measurement without complete retraining.
