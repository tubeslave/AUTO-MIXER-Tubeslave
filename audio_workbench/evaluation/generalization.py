from __future__ import annotations
def test_record(song_id:str,stages:dict,human:dict|None=None)->dict:
    return {"song_id":song_id,"stages":stages,"human":human or {},
            "required":["editing_qa","mix_critic","quality_master","loud_delivery"],
            "purpose":"cross-song generalization; never tune solely to one training song"}
def aggregate(records:list[dict])->dict:
    n=len(records);prefs=sum(1 for r in records if r.get("human",{}).get("preferred")=="candidate")
    return {"songs":n,"candidate_preferences":prefs,"preference_rate":prefs/n if n else None}
