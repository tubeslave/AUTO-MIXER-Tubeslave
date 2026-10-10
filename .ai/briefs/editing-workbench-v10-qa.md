# Editing Workbench v1.0 QA

Final QA compares raw multitrack and the cumulative edited multitrack.

Hard checks:
- exact frame count/sample rate/channel count;
- no new clipping;
- no new isolated click/pop candidates;
- no unexpected global level shift;
- re-check kick in/out and snare top/bottom phase relationship;
- inspect drum transient preservation around all local warps/gain edits;
- verify vocal pitch edits did not change total length;
- verify no edit created a discontinuity at crossfade boundaries.

If a stage fails QA, roll back that stage rather than repairing the repair.

After PASS, freeze the edited multitrack as Editing Workbench v1.0 and run the same autonomous mix,
quality mastering, and staged loud-delivery pipeline. Keep the edited premaster, quality master,
and loud master as separate artifacts.
