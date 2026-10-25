from audio_workbench.mastering.loudness_search import LoudnessSearchConfig,search

def test_search_stops_before_regression():
 base={"crest_db":18,"side_mid_db":-10,"correlation":.8}
 def render(d):
  return {"metrics":{"crest_db":18-d*.8,"side_mid_db":-10,"correlation":.8},
          "diagnostics":{"max_band_gr_db":min(1.4,d*.2),"final_max_gr_db":max(0,d-3)}}
 r=search(render,base,LoudnessSearchConfig(start_drive_db=2,step_db=.5,max_drive_db=5,max_crest_loss_db=3.5,max_final_gr_db=1.25))
 assert r["chosen"] is not None
 assert len(r["rejected"])<=1
