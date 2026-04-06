import os
import re

SUB_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Text")
OUT_FILE = os.path.join("backend", "ai", "knowledge", "youtube_mixing_tips.md")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

def clean_vtt(text):
    # Remove WEBVTT header
    text = re.sub(r'WEBVTT\n', '', text)
    # Remove timestamps
    text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n', '', text)
    # Remove tags like <c>
    text = re.sub(r'<[^>]+>', '', text)
    # Remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('Align:')]
    # Join and deduplicate repeated lines common in auto-subs
    out = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return " ".join(out)

print(f"Processing subtitle files from {SUB_DIR}...")
if not os.path.exists(SUB_DIR):
    print("Text directory does not exist yet.")
else:
    files = [f for f in os.listdir(SUB_DIR) if f.endswith('.vtt') or f.endswith('.srt')]
    
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        out.write("# Expert Mixing Tips from Video Transcripts\n\n")
        out.write("This file contains transcripts from professional mixing tutorials, providing real-world context for equalization, compression, and loudness strategies.\n\n")
        
        for f in files:
            title = f.rsplit('.', 2)[0] if f.count('.') >= 2 else f.rsplit('.', 1)[0]
            out.write(f"## Tutorial: {title}\n\n")
            with open(os.path.join(SUB_DIR, f), "r", encoding="utf-8") as in_f:
                content = clean_vtt(in_f.read())
                # chunk content into paragraphs of ~1000 chars for readability
                for i in range(0, len(content), 1000):
                    out.write(content[i:i+1000] + "...\n\n")
            out.write("\n---\n\n")
    print(f"Knowledge base updated at {OUT_FILE}")
