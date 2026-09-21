from audio_workbench.mastering.accumulation import accumulate
def test_accumulation_avoids_same_parameter_collision():
 r=accumulate({},[{"name":"a","changes":{"impact":.8}},{"name":"b","changes":{"impact":.6}},{"name":"c","changes":{"clarity":.1}}])
 assert r["components"]==["a","c"]
 assert r["requires_fresh_musical_critic"]
