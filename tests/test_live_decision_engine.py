from backend.live_runtime.contracts import ChannelFeatures,MixFeatures
from backend.live_runtime.decision_engine import propose_one

def test_masking_requires_overlap_and_low_level_evidence():
 v=ChannelFeatures(1,"Lead",-24,-8,16,activity=.9,harshness=.3)
 g=ChannelFeatures(2,"Guitar",-18,-5,13,activity=.9,harshness=.6)
 f=MixFeatures([v,g],-14,-4,10)
 h=propose_one(f,{1:"lead_vocal",2:"guitar"},{(1,2):.7})
 assert h is not None
 assert h.name=="vocal_masking_release"
 assert h.action.target=="ch:2"

def test_no_masking_action_without_overlap():
 v=ChannelFeatures(1,"Lead",-24,-8,16,activity=.9)
 g=ChannelFeatures(2,"Guitar",-18,-5,13,activity=.9)
 f=MixFeatures([v,g],-14,-4,10)
 assert propose_one(f,{1:"lead_vocal",2:"guitar"},{}) is None
