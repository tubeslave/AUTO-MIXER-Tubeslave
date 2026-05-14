import os
import datetime

OUTPUT_DIR = "mixes"

def render_mix(name="mix"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{name}_{datetime.datetime.now().strftime('%H%M%S')}.wav"
    path = os.path.join(OUTPUT_DIR, filename)

    print(f"[RENDER] Creating mix: {path}")

    with open(path, "w") as f:
        f.write("fake audio data")

    return path

if __name__ == "__main__":
    render_mix("test")
