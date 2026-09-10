"""Research-model capability registry for experimental mixing backends.

Capabilities are explicit so the pipeline never confuses a paper principle
with an actually deployed model implementation.
"""
from __future__ import annotations


def research_model_status() -> dict[str, dict[str, object]]:
    return {
        "sequential_stem_blending": {
            "principle_integrated": True,
            "paper_model_deployed": False,
            "usable_now": True,
            "mode": "orchestration_policy",
            "auto_apply": False,
            "note": "Uses growing-submix context with existing DSP/critics; not the authors' latent flow model.",
        },
        "diffvox": {
            "principle_integrated": True,
            "paper_model_deployed": False,
            "usable_now": False,
            "mode": "external_offline_backend",
            "auto_apply": False,
            "note": "Official SonyResearch code exists; deploy checkout + dependencies on a worker before use.",
        },
        "diff2mix": {
            "principle_integrated": True,
            "paper_model_deployed": False,
            "usable_now": False,
            "mode": "adapter_pending_upstream_code_checkout",
            "auto_apply": False,
            "note": "Paper/project page states code is provided, but this branch does not vendor unverified code/weights.",
        },
    }
