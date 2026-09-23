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
10. next scheduled run continues from the updated state.

Rules:
- never invent successful experiments;
- a failed experiment is useful and must be logged;
- do not make unrelated kitchen-sink changes;
- human listening remains required for subjective acceptance while the system is under development;
- no destructive live-console writes without supervised-write policy;
- no paid external generation or credit spend without explicit approval;
- model cleanup must pass RAW/CLEAN/REMOVED validation;
- keep working on another unblocked roadmap item while long compute jobs run.

Current roadmap:
1. model-based cleanup validation;
2. Perceptual Mix Critic v2;
3. Autonomous Iteration v2;
4. Mastering Director;
5. supervised live-transfer / HIL.
