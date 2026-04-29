<!-- source_type: articles -->
# Diffusion-Based Audio Inpainting

- source_type: articles
- source: arxiv
- url: http://arxiv.org/abs/2305.15266v3
- discovered_at: 2026-04-16T22:27:35.817681+00:00

## Notes

Audio inpainting aims to reconstruct missing segments in corrupted recordings. Most of existing methods produce plausible reconstructions when the gap lengths are short, but struggle to reconstruct gaps larger than about 100 ms. This paper explores recent advancements in deep learning and, particularly, diffusion models, for the task of audio inpainting. The proposed method uses an unconditionally trained generative model, which can be conditioned in a zero-shot fashion for audio inpainting, and is able to regenerate gaps of any size. An improved deep neural network architecture based on the constant-Q transform, which allows the model to exploit pitch-equivariant symmetries in audio, is also presented. The performance of the proposed algorithm is evaluated through objective and subjective metrics for the task of reconstructing short to mid-sized gaps, up to 300 ms. The results of a formal listening test show that the proposed method delivers comparable performance against the compared baselines for short gaps, such as 50 ms, while retaining a good audio quality and outperforming the baselines for wider gaps that are up to 300 ms long. The method presented in this paper can be applied to restoring sound recordings that suffer from severe local disturbances or dropouts, which must be reconstructed.
