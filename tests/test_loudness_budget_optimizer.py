from audio_workbench.mastering.loudness_budget_optimizer import SearchConfig,search
def test_optimizer_obeys_limiter_and_target():
 def ev(b):
  loud=-12+b.pregain_db*.65+b.clip_max_reduction_db*.05
  lim=max(0,b.pregain_db-4.5-b.multiband_max_gr_db*.4-b.clip_max_reduction_db*.3)
  return {"loud_section_lufs":loud,"diagnostics":{"limiter_max_gr_db":lim},"regression_pass":True}
 cfg=SearchConfig(target_loud_section_lufs=-8,tolerance_lu=.5)
 r=search(ev,cfg)
 if r["chosen"]:
  assert r["chosen"]["diagnostics"]["limiter_max_gr_db"]<=3.02
  assert abs(r["chosen"]["loud_section_lufs"]+8)<=.5
