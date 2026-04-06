import os
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed")
    exit(1)

PDF_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Text")
OUT_FILE = os.path.join("backend", "ai", "knowledge", "research_papers.md")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

if not os.path.exists(PDF_DIR):
    print("PDF directory does not exist.")
    exit(0)

pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

if not pdfs:
    print("No PDFs found.")
    exit(0)

with open(OUT_FILE, "w", encoding="utf-8") as out:
    out.write("# Research Papers on Automatic Mixing\n\n")
    out.write("This file contains text extracted from scientific papers regarding automatic multitrack mixing, DSP, and differentiable mixing consoles.\n\n")
    
    for f in pdfs:
        out.write(f"## Paper: {f}\n\n")
        pdf_path = os.path.join(PDF_DIR, f)
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                # Extract first 5 pages to save space
                if i > 4:
                    out.write("\n\n[... Remaining pages omitted ...]\n\n")
                    break
                text = page.extract_text()
                if text:
                    out.write(text + "\n\n")
        except Exception as e:
            out.write(f"Error reading {f}: {e}\n\n")
        out.write("\n---\n\n")
        
print(f"Extracted {len(pdfs)} papers into {OUT_FILE}")
