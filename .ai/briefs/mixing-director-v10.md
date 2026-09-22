# Mixing Director v1.0

First milestone deliberately uses only:
- role/context analysis,
- faders,
- pan,
- slow section-aware automation,
- diagnostic Mix Critic.

No EQ, compression, saturation, reverb, limiting or reference matching is allowed in this milestone.

Goal: prove that the system can construct a musically coherent mix from an edited multitrack before processors hide balance mistakes.

Workflow:
edited multitrack -> role inference -> arrangement density/section map -> static balance -> pan ->
bounded section automation -> render -> Mix Critic -> one bounded rebalance pass -> human A/B.

The final chorus or densest section is not automatically made louder. Energy should primarily come from arrangement,
relative foreground/background movement and later width/space stages.
