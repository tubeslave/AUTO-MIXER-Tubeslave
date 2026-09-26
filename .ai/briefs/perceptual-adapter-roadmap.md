# Perceptual adapter roadmap

Order of work:
1. deterministic Perceptual Critic (implemented);
2. artifact detector adapter;
3. music-context / section embedding adapter;
4. stem-aware foreground/background adapter;
5. fuse evidence into one bounded hypothesis;
6. render candidate;
7. automatic guardrails;
8. human level-matched A/B during development;
9. only after repeated validation may an adapter receive higher autonomy.

Important: cleanup/de-bleed evaluation remains a separate Editing concern. A perceptual mixing model must not hide source-separation artifacts by compensating for them downstream.
