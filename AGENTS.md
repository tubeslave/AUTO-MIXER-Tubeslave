# Automixer Tubeslave Agent Instructions

You are working on Dmitry's automatic live sound mixing system.

Project:
AUTO-MIXER-Tubeslave

Owner:
Dmitry is the human project owner and final decision maker.

Project goal:
Build a safe AI-assisted automatic live sound mixing system for soundchecks, rehearsals, and live concerts.

Main context:
- The system analyzes raw multichannel audio.
- It computes gain, EQ, dynamics, and balance corrections.
- It may send corrections to a real mixing console via OSC.
- The primary target console is Behringer Wing Rack.
- Live safety is more important than clever automation.

Universal safety rules for all agents:
- Do not rewrite the whole project at once.
- Do not remove working OSC, audio, analyzer, or correction code without evidence.
- Do not send uncontrolled OSC corrections to a real console.
- All console-control changes must have safety limits.
- Prefer dry-run mode before real console control.
- Prefer small, reviewable, testable changes.
- Add tests for DSP, analyzer, OSC, correction logic, and safety limits.
- Never bypass approval or sandbox settings.
- Dmitry is the final decision maker.

Recommended first task:
Analyze the AUTO-MIXER-Tubeslave repository. Do not modify files. Create a complete technical map: entry points, audio input path, analyzer modules, decision logic, OSC output path, safety limits, logging, tests, dead or duplicate code, and top 10 safe next tasks.
