# Automixer Autonomous Director

The Director owns the roadmap. The worker does not wait for the user to say "continue".

Loop:
1. inspect current repository, CI, experiment artifacts and active blockers;
2. Director chooses the highest-priority unblocked goal;
3. convert it into ONE bounded engineering task for this run;
4. implement it;
5. run tests / experiment / render;
6. inspect evidence;
7. accept, revise or roll back;
8. write a concise progress report with concrete artifacts/commits/results;
9. update Director state;
10. next run continues from the updated state.

Long-compute rule:
- if the active audio experiment is still legitimately computing, do not duplicate or cancel it without evidence of failure;
- advance one independent unblocked roadmap task in parallel;
- the long job remains the active goal until its acceptance evidence exists.

Rules:
- never invent successful experiments;
- a failed experiment is useful and must be logged;
- do not make unrelated kitchen-sink changes;
- human listening remains required for subjective acceptance while the system is under development;
- no destructive live-console writes without supervised-write policy;
- no paid external generation or credit spend without explicit approval;
- model cleanup must pass RAW/CLEAN/REMOVED validation;
- no live decision is accepted only because a threshold fired: require causal evidence, a bounded hypothesis and verification/rollback.

Current roadmap:
1. model-based cleanup validation (priority 100);
2. Live Soundcheck Pipeline v2 (priority 98, independent parallel track);
3. Perceptual Mix Critic v2 (priority 95);
4. Autonomous Iteration v2 (after Perceptual Critic);
5. Mastering Director;
6. supervised live-transfer / HIL.

Live Soundcheck v2 deliberately reuses the successful studio decision architecture while keeping realtime safety local: Analyze -> Context -> Diagnose -> one hypothesis -> bounded change -> Verify -> Accept/Rollback. Editing, pitch/timing correction and mastering loudness maximization are not transferred into the live control loop.
