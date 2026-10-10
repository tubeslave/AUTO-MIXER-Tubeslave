from audio_workbench.space_director import profile, section_plan, validate_transition, hypotheses

def test_space_profiles_are_ordered():
    d,m,l=profile("dry"),profile("medium"),profile("large")
    assert d["send_db"] < m["send_db"] < l["send_db"]
    assert d["decay_s"] < m["decay_s"] < l["decay_s"]

def test_section_plan_is_explicit():
    s=[{"name":"Verse","start_s":0,"end_s":10},{"name":"Chorus","start_s":10,"end_s":20}]
    p=section_plan(s,{"Verse":"dry","Chorus":"medium"})
    assert [x["role"] for x in p["sections"]]==["dry","medium"]
    assert validate_transition(p)["passed"]
    assert hypotheses(p)[0]["interventions"][0]["type"]=="bypass"
