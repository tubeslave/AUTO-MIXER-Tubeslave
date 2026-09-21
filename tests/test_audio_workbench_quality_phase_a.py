import pytest
from audio_workbench.song_model import build_hierarchy, section_priorities, contributors
from audio_workbench.causal import make_plan, may_auto_accept

def manifest():
    return {"tracks":[
      {"name":"Kick In","path":"k.wav","role_guess":"kick"},
      {"name":"Lead Vocal","path":"v.wav","role_guess":"vocal"},
      {"name":"GTR L","path":"g.wav","role_guess":"guitar"}],
      "sections":[{"name":"solo","role_priorities":{"GTR L":1.0,"Lead Vocal":.2}}]}

def test_hierarchy_and_section_override():
    h=build_hierarchy(manifest())
    assert h["groups"]["drums"][0]["name"]=="Kick In"
    p=section_priorities(manifest(),"solo")
    assert p["GTR L"]==1.0 and p["Lead Vocal"]==.2

def test_contributor_attribution():
    h=build_hierarchy(manifest())
    c=contributors({"Kick In":2,"Lead Vocal":1,"GTR L":7},h)
    assert c["groups"]["guitars"]["share"]==pytest.approx(.7)

def test_causal_plan_requires_no_change_and_confidence_gate():
    p=make_plan("vocal masked","guitar overlap","GTR L",
       [{"type":"eq_bell","params":{"freq_hz":3000,"db":-1,"q":1}}],
       "improve vocal clarity",["guitar_body","chorus_energy"],
       {"cause":.8,"intervention":.8})
    assert p["candidates"][0]["label"]=="no_change"
    assert not may_auto_accept(p,True,["chorus_energy"],.9)["allowed"]
    assert may_auto_accept(p,True,[],.9)["allowed"]

def test_bad_plan_rejected():
    with pytest.raises(ValueError):
        make_plan("","","x",[],"",[],{})
