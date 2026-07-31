# Avatar core quality report

The `avatar-core` batch published on 2026-07-31 was reviewed as 12-frame contact
sheets and short videos after retargeting every BVH onto the same production
VRM avatar.

Checks covered:

- first-to-last pose readability and expected action;
- limb-axis correctness after SOMA-77 to normalized-VRM retargeting;
- foot placement, ground penetration, root drift, and obvious self-collision;
- category coverage across idle, conversation, reactions, locomotion,
  swimming, transitions, object interaction, and expressive motion.

The first 30-motion pass found one prompt-level miss: `attach-backpack` looked
like a hands-on-hips gesture. Its prompt and seed were replaced and a second
bounded batch produced visible strap lifting and shoulder adjustment. Five
additional idle candidates were added, bringing the published index to 35.

`idle-shy` includes a small nervous step/cross of the feet. Treat it as an
expressive fidget rather than the default stationary idle. For a quiet default,
prefer `idle-soft-breathing`; for conversational presence, prefer
`idle-curious-listening` or `idle-thoughtful`.

The generated motions are candidates for retargeting, not facial animation.
Laugh and smile clips need a simultaneous expression/viseme layer in the host
avatar runtime. Props such as the backpack and phone likewise need host-side
attachment and contact alignment.
