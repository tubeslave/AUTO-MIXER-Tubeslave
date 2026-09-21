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
                          parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Offline plugin render. Never overwrites the source and requires subsequent verification."""
    return plugin_host.render_plugin(input_path, output_path, plugin_path, parameters)

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
