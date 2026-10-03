from audio_workbench.reference_dna import composite_reference, target_range, compare_range

def test_composite_keeps_domain_sources():
    c=composite_reference({"A":{"x":1},"B":{"x":2}},{"drums":"A","master":"B"})
    assert c["domains"]["drums"]["source"]=="A"
    assert c["domains"]["master"]["source"]=="B"

def test_reference_range_not_exact_point():
    r=target_range([-1,1],.5)
    assert compare_range(0,r)["state"]=="inside"
    assert compare_range(3,r)["state"]=="above"
