"""Assemble the deliverable package in ../Final after build.py has produced paper/main.pdf.
Usage: python package.py
Layout (same as the previous builds):
  Final/Paper_MotifFaithfulness/MotifFaithfulness.pdf, latex_source/ (standalone-compilable), latex_source.zip, cover_letter.md
  Final/README.md
"""
import os, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(HERE, "paper")
FINAL = os.path.join(ROOT, "Final")
PKG = os.path.join(FINAL, "Paper_MotifFaithfulness")
SRC_DIR = os.path.join(PKG, "latex_source")
KEEP_EXT = (".tex", ".bib", ".bst", ".pdf", ".png")


def main():
    if not os.path.exists(os.path.join(PAPER, "main.pdf")):
        sys.exit("paper/main.pdf missing: run build.py first")
    os.makedirs(PKG, exist_ok=True)
    if os.path.exists(SRC_DIR):
        shutil.rmtree(SRC_DIR)
    os.makedirs(SRC_DIR)
    for f in sorted(os.listdir(PAPER)):
        if f.endswith(KEEP_EXT) and f != "main.pdf":
            shutil.copy(os.path.join(PAPER, f), SRC_DIR)
    shutil.copy(os.path.join(PAPER, "main.pdf"), os.path.join(PKG, "MotifFaithfulness.pdf"))
    # standalone compile check of the copied sources
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=SRC_DIR, capture_output=True)
    subprocess.run(["bibtex", "main"], cwd=SRC_DIR, capture_output=True)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=SRC_DIR, capture_output=True)
    ok = os.path.exists(os.path.join(SRC_DIR, "main.pdf"))
    for f in os.listdir(SRC_DIR):
        if f.endswith((".aux", ".log", ".out", ".blg", ".toc")):
            os.remove(os.path.join(SRC_DIR, f))
    zp = os.path.join(PKG, "latex_source.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(SRC_DIR)):
            z.write(os.path.join(SRC_DIR, f), f)
    for f in ("cover_letter.md",):
        if os.path.exists(os.path.join(PAPER, f)):
            shutil.copy(os.path.join(PAPER, f), PKG)
    if os.path.exists(os.path.join(HERE, "README_final.md")):
        shutil.copy(os.path.join(HERE, "README_final.md"), os.path.join(FINAL, "README.md"))
    print("packaged ->", PKG, "| standalone compile", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
