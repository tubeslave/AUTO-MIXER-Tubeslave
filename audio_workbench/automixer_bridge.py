from __future__ import annotations

from typing import Any
from mix_agent.models import MixAction
from mix_agent.backend_bridge import MixAgentBackendBridge

def action_to_mix_action(action: dict[str,Any], target: str, action_id: str="audio-workbench") -> MixAction:
    kind=action.get("type"); p=action.get("params",{})
    if kind=="gain":
        return MixAction(id=action_id,action_type="gain_adjustment",target=target,
                         parameters={"gain_db":float(p.get("db",0.0))},
                         reason="Audio Workbench accepted offline experiment",confidence=0.5,reversible=True)
    if kind=="eq_bell":
        return MixAction(id=action_id,action_type="parametric_eq",target=target,
                         parameters={"frequency_hz":float(p["freq_hz"]),"gain_db":float(p["db"]),
                                     "q":float(p.get("q",0.8)),"band":int(p.get("band",2))},
                         reason="Audio Workbench accepted offline experiment",confidence=0.5,reversible=True)
    raise ValueError(f"action type has no safe Automixer mapping: {kind}")

def validate_for_automixer(action: dict[str,Any], target: str, channel_map: dict[str,int],
                           snapshots: dict[int,Any] | None=None) -> dict[str,Any]:
    mix_action=action_to_mix_action(action,target)
    bridge=MixAgentBackendBridge(channel_map=channel_map,snapshots=snapshots)
    return bridge.validate_or_apply([mix_action],apply=False).to_dict()
