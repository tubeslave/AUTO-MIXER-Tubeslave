from __future__ import annotations

from typing import Any
from . import core
from . import project
from . import sections as section_analysis
from . import masking
from . import compare as ab_compare
from . import experiments
from . import transients
from . import roles
from . import observers
from . import calibration
from . import plugin_host
from . import optimizer
from . import ableton
from . import renderer
from . import doctor
from . import automixer_bridge
from . import reference
from . import song_model
from . import events
from . import causal
from . import dynamic_masking
from . import macro
from . import quality_loop
from . import perceptual
from . import uncertainty
from . import preference_memory
from . import reference_dna
from . import autonomous_loop
from . import judge_calibration
from . import observer_ensemble
from . import space_director
from . import onboarding
from . import checkpoints
from . import autonomous_policy
from . import significance

try:
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("Install FastMCP or MCP Python SDK") from exc

mcp = FastMCP("Audio Workbench")

@mcp.tool()
def create_dawless_project(project_root: str, audio_dir: str, title: str = "") -> dict[str, Any]:
    """Create a DAW-independent project manifest from a folder of stems/renders."""
    return project.create_manifest(project_root, audio_dir, title)

@mcp.tool()
def get_project_context(project_root: str) -> dict[str, Any]:
    """Return the persistent DAWless project manifest."""
    return project.load_manifest(project_root)

@mcp.tool()
def set_project_context(project_root: str, sections: list[dict[str, Any]] | None = None,
                        references: list[str] | None = None,
                        notes: list[str] | None = None) -> dict[str, Any]:
    """Persist song sections, reference paths and mix-intent notes."""
    return project.update_context(project_root, sections=sections, references=references, notes=notes)





@mcp.tool()
def inspect_audio_plugin(plugin_path: str) -> dict[str, Any]:
    """Inspect a VST3/AU through Pedalboard before granting any write authority."""
    return plugin_host.inspect_plugin(plugin_path)

@mcp.tool()
def create_plugin_calibration(plugin_path: str) -> dict[str, Any]:
    """Return mandatory compatibility tests for a plugin; autonomous writes default to disabled."""
    return plugin_host.calibration_plan(plugin_path)

@mcp.tool()
def render_through_plugin(input_path: str, output_path: str, plugin_path: str,
                          parameters: dict[str, Any] | None = None,
                          calibration_record: str | None = None) -> dict[str, Any]:
    """Offline plugin render. Denied until an exact plugin build has an enabled calibration record."""
    return plugin_host.render_plugin(input_path, output_path, plugin_path, parameters,
                                     calibration_record=calibration_record)








@mcp.tool()
def assess_change_significance(delta_rms_dbfs: float, affected_fraction: float,
                               perceptual_verdict: str | None = None,
                               min_delta_rms_dbfs: float = -50.0,
                               min_affected_fraction: float = 0.02) -> dict[str, Any]:
    """Stop iterating on changes that are technically measurable but not meaningfully discriminable."""
    return significance.significance_gate(delta_rms_dbfs, affected_fraction, perceptual_verdict,
                                          min_delta_rms_dbfs, min_affected_fraction)





@mcp.tool()
def get_autonomous_mix_policy() -> dict[str, Any]:
    """Return the autonomous-first mixing policy learned from controlled mix experiments."""
    return autonomous_policy.autonomous_first_policy()

@mcp.tool()
def evaluate_dense_guitar_guard(guitar_share_db: float, section_density: float,
                                vocal_active: bool) -> dict[str, Any]:
    """Guard against over-weighting distorted guitars in dense sections."""
    return autonomous_policy.dense_guitar_guard(guitar_share_db, section_density, vocal_active)

@mcp.tool()
def record_contextual_calibration(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert a human correction into scoped evidence rather than a universal mixing rule."""
    return autonomous_policy.calibration_update(observation)

@mcp.tool()
def commit_audio_checkpoint(project_root: str, key: str, path: str,
                            inputs: dict[str, str],
                            params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist content-addressed intermediate audio so interrupted sessions can resume safely."""
    return checkpoints.commit(project_root, key, path, inputs, params)

@mcp.tool()
def check_audio_checkpoint(project_root: str, key: str,
                           inputs: dict[str, str],
                           params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify artifact bytes, dependencies and parameters before reusing a checkpoint."""
    return checkpoints.valid(project_root, key, inputs, params)

@mcp.tool()
def get_guided_mix_intake(track_count: int | None = None) -> dict[str, Any]:
    """Ask only for musical intent that cannot be safely inferred from the audio."""
    return onboarding.intake_questions(track_count)

@mcp.tool()
def compile_mix_intent(answers: dict[str, Any]) -> dict[str, Any]:
    """Compile reference roles, hierarchy, macro, space and protected objectives before autonomous mixing."""
    return onboarding.build_intent(answers)

@mcp.tool()
def build_sectional_space_plan(sections: list[dict[str, Any]],
                               role_map: dict[str, str]) -> dict[str, Any]:
    """Build dry/medium/large spatial roles by song section."""
    return space_director.section_plan(sections, role_map)

@mcp.tool()
def validate_space_transitions(plan: dict[str, Any],
                               max_send_jump_db: float = 6.0,
                               min_crossfade_ms: float = 100.0) -> dict[str, Any]:
    """Check bounded transitions before rendering sectional space automation."""
    return space_director.validate_transition(plan, max_send_jump_db, min_crossfade_ms)

@mcp.tool()
def calibrate_judge_authority(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Enable observer authority separately for each capability only after controlled trials."""
    return judge_calibration.authority_map(trials)

@mcp.tool()
def filter_judge_verdict(verdict: dict[str, Any],
                         authority: dict[str, Any]) -> dict[str, Any]:
    """Remove decision weight from perceptual axes that failed calibration."""
    return judge_calibration.filter_verdict(verdict, authority)

@mcp.tool()
def evaluate_observer_ensemble(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate calibrated observers and expose disagreement for uncertainty routing."""
    return observer_ensemble.evaluate(verdicts)

@mcp.tool()
def plan_next_autonomous_iteration(problems: list[dict[str, Any]],
                                   confidence: dict[str, float],
                                   observer_disagreement: float = 0.0) -> dict[str, Any]:
    """Choose investigation/diagnostic path without granting processing authority."""
    return autonomous_loop.next_iteration(problems, confidence, observer_disagreement)

@mcp.tool()
def should_stop_autonomous_mix(problems: list[dict[str, Any]],
                               accepted_improvements: int,
                               failed_recent_experiments: int,
                               operator_stop: bool = False) -> dict[str, Any]:
    """Apply explicit whole-song stop conditions."""
    return autonomous_loop.stop_decision(problems, accepted_improvements,
                                         failed_recent_experiments, operator_stop)

@mcp.tool()
def aggregate_blind_judges(verdicts: list[dict[str, Any]],
                           min_confidence: float = 0.65) -> dict[str, Any]:
    """Aggregate calibrated blind A/B verdicts while preserving tie/uncertain outcomes."""
    return perceptual.aggregate(verdicts, min_confidence)

@mcp.tool()
def check_perceptual_catch_trial(file_a_sha: str, file_b_sha: str,
                                 verdict: dict[str, Any]) -> dict[str, Any]:
    """Disable trust when a judge confidently prefers one of two identical files."""
    return perceptual.catch_trial_result(file_a_sha, file_b_sha, verdict)

@mcp.tool()
def assess_mix_uncertainty(confidence: dict[str, float],
                           observer_disagreement: float = 0.0) -> dict[str, Any]:
    """Route low-confidence decisions toward diagnostics, alternatives or human A/B."""
    return uncertainty.assess(confidence, observer_disagreement)

@mcp.tool()
def remember_mix_preference(project_root: str, scope: str, context: dict[str, Any],
                            intervention: dict[str, Any], result: str,
                            reason: str) -> dict[str, Any]:
    """Persist accepted, rejected, tie and uncertain decisions without making them universal."""
    return preference_memory.remember(project_root, scope, context, intervention, result, reason)

@mcp.tool()
def get_mix_preferences(project_root: str, scope: str,
                        limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent decisions only from the requested project/genre/user scope."""
    return preference_memory.retrieve(project_root, scope, limit)

@mcp.tool()
def build_composite_reference(profiles: dict[str, dict[str, Any]],
                              assignments: dict[str, str]) -> dict[str, Any]:
    """Build Reference DNA where different production domains may use different references."""
    return reference_dna.composite_reference(profiles, assignments)

@mcp.tool()
def analyze_dynamic_masking(tracks: list[dict[str, Any]],
                            priorities: dict[str, float]) -> dict[str, Any]:
    """Build time-varying masking evidence and indicate which source priority protects."""
    return dynamic_masking.dynamic_masking_graph(tracks, priorities)

@mcp.tool()
def analyze_macro_energy(audio_path: str,
                         sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure relative section energy/contrast across the whole song."""
    return macro.energy_curve(audio_path, sections)

@mcp.tool()
def check_macro_regression(before: dict[str, Any], after: dict[str, Any],
                           protected_pairs: list[dict[str, Any]],
                           tolerance_db: float = 0.75) -> dict[str, Any]:
    """Reject local improvements that collapse protected section contrast."""
    return macro.contrast_regression(before, after, protected_pairs, tolerance_db)

@mcp.tool()
def rank_next_mix_problem(problems: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prioritize the next investigation by importance, confidence, impact and uncertainty."""
    return quality_loop.rank_next_problem(problems)

@mcp.tool()
def evaluate_causal_candidate(plan: dict[str, Any], candidate: dict[str, Any],
                              target_improved: bool,
                              protected_regressions: list[str],
                              evaluation_confidence: float) -> dict[str, Any]:
    """Apply causal/protected-metric/confidence gate to one rendered candidate."""
    return quality_loop.evaluate_candidate(plan, candidate, target_improved,
                                           protected_regressions, evaluation_confidence)

@mcp.tool()
def build_song_hierarchy(project_root: str) -> dict[str, Any]:
    """Build track->group->mix hierarchy from the current project manifest."""
    return song_model.build_hierarchy(project.load_manifest(project_root))

@mcp.tool()
def get_section_priorities(project_root: str, section_name: str) -> dict[str, float]:
    """Return section-specific musical priority, with explicit context overriding role guesses."""
    return song_model.section_priorities(project.load_manifest(project_root), section_name)

@mcp.tool()
def analyze_audio_events(audio_path: str) -> dict[str, Any]:
    """Return generic onset/event timing and local crest evidence without quality labels."""
    return events.analyze_events(audio_path)

@mcp.tool()
def create_causal_experiment_plan(observation: str, hypothesis: str, target: str,
                                  interventions: list[dict[str, Any]], expected_effect: str,
                                  protected_metrics: list[str],
                                  confidence: dict[str, float] | None = None) -> dict[str, Any]:
    """Create a bounded causal plan with mandatory no-change control."""
    return causal.make_plan(observation, hypothesis, target, interventions,
                            expected_effect, protected_metrics, confidence)

@mcp.tool()
def create_reference_profile(reference_path: str,
                             sections: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create a production-trait profile. It is evidence, not a target EQ curve."""
    return reference.create_profile(reference_path, sections)

@mcp.tool()
def compare_mix_to_reference(candidate_path: str, profile: dict[str, Any],
                             candidate_sections: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compare level, dynamics, relative spectrum and stereo traits to a reference profile."""
    return reference.compare_to_reference(candidate_path, profile, candidate_sections)

@mcp.tool()
def propose_reference_hypotheses(comparison: dict[str, Any],
                                 threshold_db: float = 1.5) -> list[dict[str, Any]]:
    """Turn large reference deltas into inspection hypotheses, never direct processing commands."""
    return reference.build_hypotheses(comparison, threshold_db)

@mcp.tool()
def render_dawless_mix(project_root: str, output_path: str,
                       track_settings: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Render aligned manifest tracks to a non-destructive float stereo mix."""
    return renderer.render_mix(project_root, output_path, track_settings)

@mcp.tool()
def audio_workbench_doctor(project_root: str | None = None) -> dict[str, Any]:
    """Report runtime dependencies, project visibility, missing tracks, state DB and free disk."""
    return doctor.status(project_root)

@mcp.tool()
def validate_action_for_automixer(action: dict[str, Any], target: str,
                                  channel_map: dict[str, int]) -> dict[str, Any]:
    """Translate an accepted offline action into existing Automixer safety structures without applying it."""
    return automixer_bridge.validate_for_automixer(action, target, channel_map)

@mcp.tool()
def create_optimizer_study(project_root: str, name: str, directions: list[str],
                           metric_names: list[str]) -> dict[str, Any]:
    """Create persistent multi-objective Optuna study; optimizer is a proposer, not a judge."""
    return optimizer.create_multiobjective_study(project_root, name, directions, metric_names)

@mcp.tool()
def get_pareto_candidates(project_root: str, name: str) -> list[dict[str, Any]]:
    """Return non-dominated optimization candidates for separate musical evaluation."""
    return optimizer.pareto_trials(project_root, name)

@mcp.tool()
def ableton_set_track_volume(track_id: int, value: float,
                             host: str = "127.0.0.1", send_port: int = 11000) -> dict[str, Any]:
    """Optional AbletonOSC write. Must be followed by readback/render verification."""
    return ableton.set_track_volume(track_id, value, host, send_port)

@mcp.tool()
def ableton_set_device_parameter(track_id: int, device_id: int, parameter_id: int, value: float,
                                 host: str = "127.0.0.1", send_port: int = 11000) -> dict[str, Any]:
    """Optional AbletonOSC parameter write. Must be followed by readback/render verification."""
    return ableton.set_device_parameter(track_id, device_id, parameter_id, value, host, send_port)

_qwen_observer = None

@mcp.tool()
def qwen_audio_observe(audio_path: str, question: str, start_s: float = 0.0,
                       end_s: float | None = None) -> dict[str, Any]:
    """Ask local Qwen2-Audio about <=30 s mono/16 kHz evidence. Never treats prose as a mix verdict."""
    global _qwen_observer
    if _qwen_observer is None:
        _qwen_observer = observers.Qwen2AudioObserver()
    return _qwen_observer.ask(audio_path, question, start_s, end_s)

@mcp.tool()
def audiobox_aesthetics(audio_path: str, start_s: float | None = None,
                        end_s: float | None = None) -> dict[str, Any]:
    """Return Audiobox CE/CU/PC/PQ as secondary evidence/regression flags."""
    return observers.audiobox_scores(audio_path, start_s, end_s)

@mcp.tool()
def muq_mulan_research_similarity(audio_path: str, texts: list[str],
                                  allow_noncommercial: bool = False,
                                  device: str = "cpu") -> dict[str, Any]:
    """Research-only MuQ-MuLan similarity. License guard defaults to deny."""
    return observers.muq_mulan_similarity(audio_path, texts, allow_noncommercial, device)

@mcp.tool()
def create_observer_calibration(project_root: str, clean_path: str,
                                perturbations: list[dict[str, Any]]) -> dict[str, Any]:
    """Create capability-by-capability observer exam before learned evidence gets decision weight."""
    return calibration.make_observer_exam(project_root, clean_path, perturbations)

@mcp.tool()
def analyze_transients(audio_path: str) -> dict[str, Any]:
    """Detect transient events and timing statistics without judging their musical quality."""
    return transients.analyze_transients(audio_path)

@mcp.tool()
def get_role_priorities(project_root: str, section_name: str | None = None) -> dict[str, Any]:
    """Return role priors for contextual reasoning; explicit project intent overrides filename guesses."""
    return roles.role_priorities(project.load_manifest(project_root), section_name)

@mcp.tool()
def run_audio_experiment(project_root: str, input_path: str, hypothesis: str,
                         actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Render reversible bypass/gain/EQ/compression candidates for one explicit hypothesis."""
    return experiments.run_experiment(project_root, input_path, hypothesis, actions)

@mcp.tool()
def select_experiment_candidate(project_root: str, experiment_id: str, candidate: int,
                                reason: str, evaluator: str = "human_or_calibrated_judge") -> dict[str, Any]:
    """Commit an evaluated candidate to the decision log. Does not overwrite source audio."""
    return experiments.choose_candidate(project_root, experiment_id, candidate, reason, evaluator)

@mcp.tool()
def analyze_sections(audio_path: str, sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure one render by musical sections supplied by the project context."""
    return {"sections": section_analysis.section_features(audio_path, sections)}

@mcp.tool()
def find_suspicious_windows(audio_path: str, window_s: float = 8.0,
                            hop_s: float = 4.0, top_k: int = 8) -> dict[str, Any]:
    """Find statistically unusual windows for extra inspection; never labels them bad."""
    return section_analysis.find_outlier_windows(audio_path, window_s, hop_s, top_k)

@mcp.tool()
def analyze_masking(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build pairwise time-frequency overlap graph. It is diagnostic, not an EQ command."""
    return masking.masking_graph(tracks)

@mcp.tool()
def compare_renders(a_path: str, b_path: str, loudness_match: bool = True) -> dict[str, Any]:
    """Compare two renders without converting difference magnitude into preference."""
    return ab_compare.compare(a_path, b_path, loudness_match)

@mcp.tool()
def make_blind_ab(a_path: str, b_path: str, salt: str) -> dict[str, str]:
    """Return deterministic blinded A/B assignment for evaluation."""
    return ab_compare.blind_labels(a_path, b_path, salt)

@mcp.tool()
def inspect_audio(project_root: str, audio_path: str) -> dict[str, Any]:
    """Register immutable audio identity and return deterministic signal analysis."""
    identity = core.register(project_root, audio_path)
    return {"identity": identity, "analysis": core.analyze(audio_path)}

@mcp.tool()
def analyze_audio(audio_path: str, include_loudness: bool = True) -> dict[str, Any]:
    """Analyze signal. Metrics are evidence; this tool never claims artistic quality."""
    out = core.analyze(audio_path)
    if include_loudness:
        out["ffmpeg_ebur128"] = core.ffmpeg_loudness(audio_path)
    return out

@mcp.tool()
def record_check(project_root: str, render_sha: str, check_name: str,
                 result: dict[str, Any], status: str = "ok") -> dict[str, Any]:
    """Attach a completed check to one exact render."""
    return core.record_check(project_root, render_sha, check_name, result, status)

@mcp.tool()
def invalidate_after_change(project_root: str, render_sha: str, change_type: str) -> dict[str, Any]:
    """Mark dependent checks stale after a proposed or applied change."""
    return core.invalidate(project_root, render_sha, change_type)

@mcp.tool()
def get_coverage(project_root: str, render_sha: str) -> dict[str, Any]:
    """Return mandatory-check coverage. Finalization is blocked by missing/stale checks."""
    return core.coverage(project_root, render_sha)

@mcp.tool()
def get_next_task(project_root: str, render_sha: str) -> dict[str, Any]:
    """Return the next mandatory verification task."""
    return core.next_task(project_root, render_sha)

@mcp.tool()
def record_decision(project_root: str, render_sha: str, hypothesis: str,
                    action: dict[str, Any], outcome: str = "proposed") -> dict[str, Any]:
    """Persist hypothesis -> action -> outcome for later audit/learning."""
    return core.log_decision(project_root, render_sha, hypothesis, action, outcome)

if __name__ == "__main__":
    mcp.run()
