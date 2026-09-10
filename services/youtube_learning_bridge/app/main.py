from __future__ import annotations
import os,re,json,uuid
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import httpx
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import HTMLResponse
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel

app=FastAPI(title='YouTube Learning Bridge',version='0.1.0')
DATA=Path(os.getenv('DATA_DIR','./data')); (DATA/'uploads').mkdir(parents=True,exist_ok=True)
DB=DATA/'youtube_learning.jsonl'
MODEL=None
PATTERNS={
 'eq':r'\b(eq|equaliz|эквал|частот|hz|khz)\b',
 'compression':r'\b(compress|компресс|ratio|threshold|attack|release|1176|la-2a|la2a|la-3a)\b',
 'reverb':r'\b(reverb|реверб|room|plate|hall|ambience|chamber|predelay|decay)\b',
 'delay':r'\b(delay|slap|echo|дилей|задержк)\b',
 'saturation':r'\b(saturat|distort|clip|drive|harmonic|сатурац|клип)\b',
 'transient':r'\b(transient|punch|панч|crest|крэст|envelope|sustain)\b',
 'stereo':r'\b(stereo|width|pan|mono|стерео|панорам)\b',
 'loudness':r'\b(lufs|loudness|true peak|dbtp|громкост)\b',
 'drums':r'\b(kick|snare|tom|overhead|drum|барабан|малый|бочка)\b',
 'bass':r'\b(bass|бас)\b','guitars':r'\b(guitar|гитар)\b','vocals':r'\b(vocal|voice|вокал|голос)\b'
}
NUM=re.compile(r'(?<!\w)-?\d+(?:\.\d+)?\s?(?:dBFS|dBTP|dB|LUFS|Hz|kHz|ms|s|sec|:1|%)',re.I)

def vid(x:str)->str:
    x=x.strip()
    if re.fullmatch(r'[A-Za-z0-9_-]{11}',x): return x
    p=urlparse(x); h=(p.hostname or '').lower()
    if h in {'youtu.be','www.youtu.be'}:
        v=p.path.strip('/').split('/')[0]
        if re.fullmatch(r'[A-Za-z0-9_-]{11}',v): return v
    if 'youtube.com' in h:
        if p.path=='/watch':
            v=parse_qs(p.query).get('v',[None])[0]
            if v and re.fullmatch(r'[A-Za-z0-9_-]{11}',v): return v
        ps=[a for a in p.path.split('/') if a]
        if len(ps)>1 and ps[0] in {'shorts','embed','live'} and re.fullmatch(r'[A-Za-z0-9_-]{11}',ps[1]): return ps[1]
    raise ValueError('Invalid YouTube URL/video ID')

async def meta(x:str):
    v=vid(x); url=f'https://www.youtube.com/watch?v={v}'; key=os.getenv('YOUTUBE_API_KEY')
    if key:
        async with httpx.AsyncClient(timeout=20) as c:
            r=await c.get('https://www.googleapis.com/youtube/v3/videos',params={'part':'snippet,contentDetails','id':v,'key':key})
            r.raise_for_status(); a=r.json().get('items',[])
        if a:
            s=a[0].get('snippet',{}); d=a[0].get('contentDetails',{})
            return {'video_id':v,'url':url,'title':s.get('title'),'channel':s.get('channelTitle'),'description':s.get('description'),'duration':d.get('duration'),'metadata_method':'youtube_data_api_v3'}
    async with httpx.AsyncClient(timeout=20) as c:
        r=await c.get('https://www.youtube.com/oembed',params={'url':url,'format':'json'}); r.raise_for_status(); o=r.json()
    return {'video_id':v,'url':url,'title':o.get('title'),'channel':o.get('author_name'),'metadata_method':'youtube_oembed'}

def public_transcript(x:str,languages=('ru','en')):
    v=vid(x); api=YouTubeTranscriptApi(); ls=api.list(v); chosen=None
    for lang in languages:
        try: chosen=ls.find_transcript([lang]); break
        except Exception: pass
    if chosen is None:
        arr=list(ls)
        if not arr: raise RuntimeError('No exposed transcript')
        manual=[t for t in arr if not t.is_generated]; chosen=manual[0] if manual else arr[0]
    got=chosen.fetch()
    seg=[{'start':float(s.start),'duration':float(s.duration),'text':str(s.text).strip()} for s in got if str(s.text).strip()]
    return {'video_id':v,'language':getattr(chosen,'language_code',None),'is_generated':getattr(chosen,'is_generated',None),'method':'youtube_transcript_api_third_party','segments':seg,'text':' '.join(s['text'] for s in seg),'limitations':['No media stream downloaded.','Transcript availability depends on YouTube/caption settings.']}

def mentions(tr):
    out=[]; ss=tr['segments']
    for i,s in enumerate(ss):
        cats=[k for k,p in PATTERNS.items() if re.search(p,s['text'],re.I)]
        if cats:
            lo=max(0,i-1); hi=min(len(ss),i+2); tx=' '.join(z['text'] for z in ss[lo:hi])
            out.append({'start':ss[lo]['start'],'end':ss[hi-1]['start']+ss[hi-1]['duration'],'categories':cats,'text':tx,'numbers':NUM.findall(tx)})
    return out

def save(row):
    DB.parent.mkdir(parents=True,exist_ok=True)
    with DB.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')

def whisper():
    global MODEL
    if MODEL is None:
        MODEL=WhisperModel(os.getenv('WHISPER_MODEL','small'),device=os.getenv('WHISPER_DEVICE','cpu'),compute_type=os.getenv('WHISPER_COMPUTE_TYPE','int8'))
    return MODEL

@app.get('/healthz')
def health(): return {'ok':True,'version':'0.1.0','whisper_model':os.getenv('WHISPER_MODEL','small')}

@app.get('/')
def home():
    return HTMLResponse("""<meta name=viewport content='width=device-width,initial-scale=1'><style>body{font-family:-apple-system;max-width:800px;margin:30px auto;padding:16px}input,button{font-size:16px;padding:10px;margin:5px 0;width:100%}pre{white-space:pre-wrap;background:#eee;padding:12px}</style><h1>YouTube Learning Bridge</h1><input id=u placeholder='YouTube URL'><button onclick='go()'>Получить транскрипт</button><pre id=o></pre><form id=f><input type=file id=x accept='audio/*,video/*'><input id=l placeholder='ru / en, необязательно'><button>Транскрибировать файл</button></form><pre id=p></pre><script>async function go(){o.textContent='...';let r=await fetch('/api/ingest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u.value})});o.textContent=JSON.stringify(await r.json(),null,2)}f.onsubmit=async e=>{e.preventDefault();p.textContent='...';let d=new FormData();d.append('file',x.files[0]);d.append('language',l.value);let r=await fetch('/api/upload',{method:'POST',body:d});p.textContent=JSON.stringify(await r.json(),null,2)}</script>""")

@app.post('/api/ingest')
async def ingest(body:dict):
    m=await meta(body['url'])
    try:
        tr=public_transcript(body['url'],tuple(body.get('languages') or ['ru','en']))
        row={'metadata':m,'transcript':tr,'mixing_mentions':mentions(tr),'status':'transcript_ready'}
    except Exception as e:
        row={'metadata':m,'transcript':None,'mixing_mentions':[],'status':'needs_audio_or_transcript','reason':str(e)}
    save(row); return row

@app.post('/api/upload')
async def upload(file:UploadFile=File(...),language:str=Form('')):
    ext=Path(file.filename or '').suffix.lower()
    if ext not in {'.wav','.mp3','.m4a','.aac','.flac','.ogg','.opus','.mp4','.mov','.mkv','.webm'}: raise HTTPException(400,'Unsupported file type')
    p=DATA/'uploads'/f'{uuid.uuid4().hex}{ext}'; maxb=int(os.getenv('MAX_UPLOAD_MB','500'))*1024*1024; size=0
    try:
        with p.open('wb') as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>maxb: raise HTTPException(413,'File too large')
                f.write(chunk)
        segs,info=whisper().transcribe(str(p),language=language or None,beam_size=5,vad_filter=True)
        ss=[{'start':float(s.start),'duration':float(s.end-s.start),'text':(s.text or '').strip()} for s in segs if (s.text or '').strip()]
        tr={'video_id':'uploaded_media','language':getattr(info,'language',None),'is_generated':True,'method':'faster_whisper_uploaded_media','segments':ss,'text':' '.join(s['text'] for s in ss),'limitations':['Verify plugin names and numeric settings against source.']}
        row={'uploaded_media':file.filename,'transcript':tr,'mixing_mentions':mentions(tr),'status':'transcript_ready'}
        save(row); return row
    finally:
        p.unlink(missing_ok=True)
