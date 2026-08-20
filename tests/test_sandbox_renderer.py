import numpy as np
import pytest

from ai_mixing_pipeline.models import CandidateAction, MixCandidate
from ai_mixing_pipeline.sandbox_renderer import SandboxRenderer
from ai_mixing_pipeline.safety_governor import SafetyGovernor
from ai_mixing_pipeline.decision_layer.action_schema import CandidateActionSet, GainAction, NoChangeAction
from ai_mixing_pipeline.decision_layer.fallback_virtual_mixer import FallbackVirtualMixer
from ai_mixing_pipeline.decision_layer.sandbox_renderer import DecisionSandboxRenderer


def _sine(freq, amp=0.2, sr=48000):
    t = np.arange(int(sr * 0.2), dtype=np.float32) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)[:, None]


def test_sandbox_renderer_creates_initial_and_candidate_without_clipping(tmp_path):
    stems = {
        "01_kick": _sine(60.0, 0.2),
        "02_vocal": _sine(880.0, 0.1),
    }
    roles = {"01_kick": "kick", "02_vocal": "lead_vocal"}
    renderer = SandboxRenderer({"safety": {"max_true_peak_dbfs": -1.0}})

    initial = renderer.render_initial_mix(stems, roles, 48000, tmp_path / "000_initial_mix.wav")
    candidate = MixCandidate(
        candidate_id="001_candidate_gain_balance",
        label="gain_balance",
        render_filename="001_candidate_gain_balance.wav",
        actions=[
            CandidateAction(
                action_type="gain_change",
                target="02_vocal",
                parameters={"gain_db": 0.5},
            )
        ],
    )
    rendered = renderer.render_candidate(
        candidate,
        stems,
        roles,
        48000,
        tmp_path,
        target_lufs=initial.metrics["level"]["integrated_lufs"],
        loudness_match=True,
    )

    assert (tmp_path / "000_initial_mix.wav").exists()
    assert (tmp_path / "001_candidate_gain_balance.wav").exists()
    assert rendered.loudness_matched is True
    assert rendered.metrics["level"]["clip_count"] == 0
    assert rendered.metrics["level"]["true_peak_dbtp"] <= -1.0
    assert rendered.metadata["render_peak_ceiling_dbfs"] == -1.6


def test_sandbox_renderer_pre_trims_candidates_before_safety(tmp_path):
    stems = {
        "01_kick": _sine(60.0, 0.9),
        "02_vocal": _sine(880.0, 0.9),
    }
    roles = {"01_kick": "kick", "02_vocal": "lead_vocal"}
    config = {
        "safety": {"max_true_peak_dbfs": -1.0, "min_headroom_db": 1.0},
        "offline_test": {"safe_render_peak_margin_db": 0.6},
    }
    renderer = SandboxRenderer(config)
    candidate = MixCandidate(
        candidate_id="001_candidate_gain_balance",
        label="gain_balance",
        render_filename="001_candidate_gain_balance.wav",
        actions=[
            CandidateAction(
                action_type="gain_change",
                target="02_vocal",
                parameters={"gain_db": 0.5},
            )
        ],
    )

    rendered = renderer.render_candidate(
        candidate,
        stems,
        roles,
        48000,
        tmp_path,
        loudness_match=False,
    )
    safety = SafetyGovernor(config).evaluate(candidate, rendered.path)

    assert rendered.output_gain_db < 0.0
    assert rendered.metrics["level"]["true_peak_dbtp"] <= -1.0
    assert safety.passed is True


@pytest.mark.parametrize(
    ("candidate_id", "render_filename"),
    [
        ("000_initial_mix", "../escaped-initial.wav"),
        ("candidate_escape", "../escaped-explicit.wav"),
        ("../escaped-fallback", ""),
        ("candidate_absolute", "/tmp/escaped-absolute.wav"),
    ],
)
def test_sandbox_renderer_rejects_candidate_paths_outside_render_root(
    tmp_path,
    candidate_id,
    render_filename,
):
    renderer = SandboxRenderer()
    candidate = MixCandidate(
        candidate_id=candidate_id,
        label="escape",
        render_filename=render_filename,
    )

    with pytest.raises(ValueError, match="Candidate render filename"):
        renderer.render_candidate(
            candidate,
            {"stem": _sine(440.0)},
            {"stem": "unknown"},
            48000,
            tmp_path / "renders",
            loudness_match=False,
        )

    assert not (tmp_path / "escaped-initial.wav").exists()
    assert not (tmp_path / "escaped-explicit.wav").exists()
    assert not (tmp_path / "escaped-fallback.wav").exists()


def test_sandbox_renderer_rejects_symlinked_candidate_outside_render_root(tmp_path):
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"preserve-me")
    (output_dir / "candidate.wav").symlink_to(outside)
    renderer = SandboxRenderer()
    candidate = MixCandidate(
        candidate_id="candidate_symlink",
        label="escape",
        render_filename="candidate.wav",
    )

    with pytest.raises(ValueError, match="escapes the render directory"):
        renderer.render_candidate(
            candidate,
            {"stem": _sine(440.0)},
            {"stem": "unknown"},
            48000,
            output_dir,
            loudness_match=False,
        )

    assert outside.read_bytes() == b"preserve-me"


def test_decision_layer_sandbox_renderer_creates_manifest(tmp_path):
    multitrack = tmp_path / "multitrack"
    multitrack.mkdir()
    import soundfile as sf

    sf.write(multitrack / "vocal.wav", _sine(440.0, 0.2), 48000)
    mixer = FallbackVirtualMixer({"sample_rate": 48000, "safety": {"max_true_peak_dbfs": -1.0}})
    mixer.load_project(multitrack)
    renderer = DecisionSandboxRenderer(
        mixer,
        tmp_path / "run",
        {"loudness_match": True, "safety": {"max_true_peak_dbfs": -1.0}},
    )

    results = renderer.render_candidates(
        [
            CandidateActionSet("candidate_000_no_change", [NoChangeAction()]),
            CandidateActionSet("candidate_001_vocal_up", [GainAction("vocal", 0.5)]),
        ]
    )

    assert (tmp_path / "run" / "reports" / "candidate_manifest.json").exists()
    assert (tmp_path / "run" / "renders" / "candidate_000_no_change.wav").exists()
    assert set(results) == {"candidate_000_no_change", "candidate_001_vocal_up"}
