from audio_workbench.onboarding import intake_questions, build_intent

def test_intake_requests_musical_context_not_redundant_audio_facts():
    q=intake_questions(48)
    ids={x["id"] for x in q["questions"]}
    assert {"references","hierarchy","energy","space","character","do_not_break"} <= ids

def test_required_intent_blocks_readiness():
    r=build_intent({"hierarchy":"vocal verse, guitar solo"})
    assert not r["ready"] and "space" in r["missing"]

def test_reference_domains_are_explicit():
    a={"hierarchy":"x","energy":"x","space":"x","character":["punch"],"do_not_break":["groove"],
       "references":[{"name":"Ref","domains":["drums","master"]}]}
    r=build_intent(a)
    assert r["ready"] and r["references"][0]["domains"]==["drums","master"]
