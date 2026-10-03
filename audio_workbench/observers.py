from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def _clip(path: str, start_s: float, end_s: float | None, max_s: float) -> tuple[Any,int]:
    import librosa
    duration = max_s if end_s is None else min(max_s, max(0.1, end_s-start_s))
    y,sr = librosa.load(path, sr=16000, mono=True, offset=max(0,start_s), duration=duration)
    return y,sr

class Qwen2AudioObserver:
    """Lazy local observer. Its text is evidence, never an artistic verdict."""
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    max_clip_s = 30.0

    def __init__(self, device: str = "auto"):
        self.device=device
        self._model=None; self._processor=None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self._processor=AutoProcessor.from_pretrained(self.model_id)
        kwargs={"device_map":self.device}
        if torch.backends.mps.is_available():
            # device_map auto support varies on Apple Silicon; explicit MPS fallback is safer.
            kwargs={}
        self._model=Qwen2AudioForConditionalGeneration.from_pretrained(self.model_id,**kwargs)
        if torch.backends.mps.is_available():
            self._model=self._model.to("mps")
        self._model.eval()

    def ask(self, audio_path: str, question: str, start_s: float=0.0,
            end_s: float | None=None, max_new_tokens: int=180) -> dict[str,Any]:
        self._load()
        import torch
        audio,sr=_clip(audio_path,start_s,end_s,self.max_clip_s)
        conversation=[{"role":"user","content":[
            {"type":"audio","audio_url":"local"},
            {"type":"text","text":question}
        ]}]
        text=self._processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
        inputs=self._processor(text=text,audios=[audio],sampling_rate=sr,return_tensors="pt",padding=True)
        device=next(self._model.parameters()).device
        inputs={k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            ids=self._model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        ids=ids[:,inputs["input_ids"].shape[1]:]
        answer=self._processor.batch_decode(ids,skip_special_tokens=True)[0]
        return {
          "model":self.model_id,"answer":answer,"start_s":start_s,
          "end_s":min(start_s+self.max_clip_s,end_s) if end_s else start_s+self.max_clip_s,
          "input_sample_rate":16000,
          "limitations":[
            "observer input is mono 16 kHz and clips are capped at 30 s",
            "not valid for full-band air/stereo judgments",
            "answer is a hypothesis requiring deterministic or A/B verification"
          ]
        }

def audiobox_scores(audio_path: str, start_s: float | None=None,
                    end_s: float | None=None) -> dict[str,Any]:
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install audiobox_aesthetics") from exc
    predictor=initialize_predictor()
    item={"path":audio_path}
    if start_s is not None: item["start_time"]=start_s
    if end_s is not None: item["end_time"]=end_s
    result=predictor.forward([item])[0]
    return {
      "model":"facebook/audiobox-aesthetics","scores":result,
      "limitations":[
        "CE/CU/PC/PQ are model axes, not percentages or a single mix-quality objective",
        "use as regression flags/evidence; never auto-maximize"
      ]
    }

def muq_mulan_similarity(audio_path: str, texts: list[str],
                         allow_noncommercial: bool=False, device: str="cpu") -> dict[str,Any]:
    if not allow_noncommercial:
        raise PermissionError(
          "MuQ-MuLan checkpoint is CC-BY-NC-4.0. Set allow_noncommercial=true only for a research/non-commercial run."
        )
    import librosa, torch
    try:
        from muq import MuQMuLan
    except ImportError as exc:
        raise RuntimeError("Install research-only optional dependency: pip install muq") from exc
    wav,_=librosa.load(audio_path,sr=24000,mono=True)
    model=MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()
    x=torch.tensor(wav,dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ae=model(wavs=x)
        te=model(texts=texts)
        sim=model.calc_similarity(ae,te)
    vals=sim.detach().cpu().numpy().reshape(-1).tolist()
    return {
      "model":"OpenMuQ/MuQ-MuLan-large","texts":texts,"similarity":vals,
      "input_sample_rate":24000,"license_guard":"CC-BY-NC-4.0 research/non-commercial only",
      "limitations":["text similarity is not mix quality and must not directly move faders/EQ"]
    }
