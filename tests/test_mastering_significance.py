from audio_workbench.mastering.significance import classify

def test_inaudible_safe_change_is_delivery_only():
 r=classify(False,{"failures":[]},1.3)
 assert r["class"]=="delivery_variant" and not r["promote_quality_baseline"]

def test_regression_rejected():
 r=classify(True,{"failures":["kick_attack_loss"]},1.0)
 assert r["class"]=="reject"
