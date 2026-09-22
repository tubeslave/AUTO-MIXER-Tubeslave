from __future__ import annotations
ALLOWED=("set_gain_db","set_pan","set_eq_band","set_compressor","set_send_db","write_automation","bypass")
def validate(actions:list[dict])->dict:
    fail=[]
    for i,a in enumerate(actions):
        if a.get("action") not in ALLOWED:fail.append({"index":i,"reason":"action_not_allowed"})
        if a.get("action")=="set_gain_db" and abs(float(a.get("value",0)))>18:fail.append({"index":i,"reason":"gain_out_of_bounds"})
    return {"accept":not fail,"failures":fail}
def supervised(actions:list[dict])->dict:
    v=validate(actions);return {**v,"requires_commit":True,"dry_run_first":True}
