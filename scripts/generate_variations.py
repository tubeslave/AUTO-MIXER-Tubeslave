from render_mix import render_mix

def generate():
    variations = ["vocal_forward", "balanced", "aggressive"]
    results = []

    for v in variations:
        path = render_mix(v)
        results.append(path)

    print("\\nGenerated mixes:")
    for r in results:
        print(r)

if __name__ == "__main__":
    generate()
