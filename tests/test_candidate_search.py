from audio_workbench.mastering.candidate_search import Candidate,search

def test_rejects_failed_critic_and_shortlists_survivors():
 base={"crest_db":17.0}
 cs=(Candidate("ok",{},"x"),Candidate("bad",{},"y"))
 def render(c):
  good=c.name=="ok"
  return {"metrics":{"crest_db":16.8},
   "critic":{"accept":good,"spectral_shift":{"rms_shift_db":.1},
             "low_end_punch":{"median_punch_change_db":-.05}}}
 r=search(render,base,cs)
 assert r["rows"][1]["status"]=="reject"
 assert r["shortlist"][0]["candidate"]["name"]=="ok"
