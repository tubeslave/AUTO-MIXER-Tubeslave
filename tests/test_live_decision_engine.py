from backend.live_runtime.contracts import ChannelFeatures,EqBandLocator,MixFeatures
from backend.live_runtime.decision_engine import propose_one

def test_masking_requires_overlap_and_explicit_physical_eq_locator():
 v=ChannelFeatures(1,"Lead",-24,-8,16,activity=.9,harshness=.3)
 g=ChannelFeatures(2,"Guitar",-18,-5,13,activity=.9,harshness=.6)
 f=MixFeatures([v,g],-14,-4,10)
 locator=EqBandLocator(3,3200.0,1.4)
 h=propose_one(
  f,{1:"lead_vocal",2:"guitar"},{(1,2):.7},
  {(2,"vocal_masking"):locator},
 )
 assert h is not None
 assert h.name=="vocal_masking_release"
 assert h.action.target=="ch:2"
 assert h.action.parameter=="eq_gain_delta_db"
 assert h.action.value==-0.7
 assert h.action.eq_locator==locator

def test_masking_hypothesis_is_not_actionable_without_eq_locator():
 v=ChannelFeatures(1,"Lead",-24,-8,16,activity=.9,harshness=.3)
 g=ChannelFeatures(2,"Guitar",-18,-5,13,activity=.9,harshness=.6)
 f=MixFeatures([v,g],-14,-4,10)
 assert propose_one(f,{1:"lead_vocal",2:"guitar"},{(1,2):.7}) is None

def test_no_masking_action_without_overlap():
 v=ChannelFeatures(1,"Lead",-24,-8,16,activity=.9)
 g=ChannelFeatures(2,"Guitar",-18,-5,13,activity=.9)
 f=MixFeatures([v,g],-14,-4,10)
 locator=EqBandLocator(3,3200.0,1.4)
 assert propose_one(
  f,{1:"lead_vocal",2:"guitar"},{},
  {(2,"vocal_masking"):locator},
 ) is None

def test_harshness_requires_locator_and_uses_relative_gain():
 g=ChannelFeatures(2,"Guitar",-18,-5,13,activity=.9,harshness=.9)
 f=MixFeatures([g],-14,-4,10)
 assert propose_one(f,{2:"guitar"}) is None
 locator=EqBandLocator(4,4100.0,1.8)
 h=propose_one(f,{2:"guitar"},eq_locators={(2,"harshness"):locator})
 assert h is not None
 assert h.name=="source_harshness"
 assert h.action.parameter=="eq_gain_delta_db"
 assert h.action.value==-0.6
 assert h.action.eq_locator==locator
