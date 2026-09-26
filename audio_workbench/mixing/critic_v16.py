from __future__ import annotations
def evaluate(m):
    fail=[]
    if not (-4.0<=m["vocal_to_music_db"]<=3.0):fail.append("vocal_balance")
    if not (-5.0<=m["kick_to_bass_db"]<=5.0):fail.append("kick_bass_balance")
    if m["peak_dbfs"]>-2.0:fail.append("headroom")
    if m["crest_db"]<10.0:fail.append("overcompressed")
    if m["side_mid_db"]>-5.0:fail.append("excess_width")
    return {"accept":not fail,"failures":fail}
def should_iterate(result):
    return not result["accept"]
