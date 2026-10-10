from __future__ import annotations
from typing import Any

def autonomous_first_policy() -> dict[str,Any]:
    return {
      "default_mode":"autonomous_first",
      "inputs":["multitrack_audio","track_names"],
      "initially_forbidden":["reference_targets","guided_mix_intent","historical_user_preferences"],
      "stages":[
        "technical_preflight",
        "role_and_section_inference",
        "autonomous_balance",
        "autonomous_processing",
        "self_critique",
        "optional_external_constraints",
        "bounded_revision"
      ],
      "principle":"References and user intent are optional priors after a stable autonomous baseline, not mandatory targets."
    }

def dense_guitar_guard(guitar_share_db: float, section_density: float,
                       vocal_active: bool, threshold_db: float=-4.5) -> dict[str,Any]:
    # Learned from Premiera calibration: dense distorted guitars were systematically over-weighted.
    risk = section_density >= .65 and guitar_share_db > threshold_db
    correction = -1.5 if risk else 0.0
    if risk and vocal_active: correction -= .5
    return {"risk":risk,"suggested_delta_db":correction,
            "max_auto_delta_db":-2.0,
            "requires_regression":["drum_punch","low_end_foundation","vocal_intelligibility"]}

def calibration_update(observation: dict[str,Any]) -> dict[str,Any]:
    """Turn human corrections into bounded policy evidence, never universal truth."""
    return {
      "scope":observation.get("scope","song"),
      "feature_context":observation.get("feature_context",{}),
      "preferred_delta_db":observation.get("preferred_delta_db"),
      "confidence":observation.get("confidence","single_song"),
      "generalization":"candidate_prior_only_until_repeated_across_songs"
    }

def stop_rule(history: list[dict[str,Any]]) -> dict[str,Any]:
    material=[h for h in history if abs(float(h.get("delta_db",0))) >= .5]
    if history and not material:
        return {"stop":True,"reason":"remaining changes are below useful calibration scale"}
    return {"stop":False}
