from __future__ import annotations
from dataclasses import dataclass
from .contracts import LiveMode

@dataclass(frozen=True)
class LiveIntent:
    command:str
    scope:str|None=None
    amount:str|None=None
    requested_mode:LiveMode|None=None

ALIASES={
 "начинай саундчек":"start_soundcheck",
 "начать саундчек":"start_soundcheck",
 "только барабаны":"scope_drums",
 "проанализируй весь микс":"analyze_mix",
 "покажи изменения":"show_proposals",
 "применяй только безопасные изменения":"auto_safe",
 "отмени последний проход":"undo_last_pass",
 "ничего больше не меняй":"freeze",
 "стоп":"freeze",
}

def parse_fixed_command(text:str)->LiveIntent|None:
    """Safety-critical voice commands use an explicit vocabulary before any LLM fallback."""
    q=" ".join(text.lower().strip().split())
    cmd=ALIASES.get(q)
    if not cmd:return None
    mode={"auto_safe":LiveMode.AUTO_SAFE,"freeze":LiveMode.FREEZE}.get(cmd)
    return LiveIntent(cmd,requested_mode=mode)
