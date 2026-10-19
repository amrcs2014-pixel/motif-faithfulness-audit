"""Rebuild the paper end to end from the finished runs: aggregate results, regenerate numbers, tables and figures,
compile paper/main.pdf, and check for unresolved macros or references.

Usage: python build.py            (full: analyze -> figs_tables -> pdflatex/bibtex)
       python build.py --tex-only (skip the analysis; recompile only)
Stages before this one (each resumable, run once):  src/fetch_smiles.py -> src/prep_data.py -> src/chem.py ->
src/splits.py -> scheduled queues src/gpu_queue.py (train.py / explain.py) and src/cpu_jobs.py (baselines.py).
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src"); PAPER = os.path.join(HERE, "paper")


def run(cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode


def compile_tex(stem="main"):
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], cwd=PAPER)
    run(["bibtex", stem], cwd=PAPER)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], cwd=PAPER)


def check(stem="main"):
    log = open(os.path.join(PAPER, f"{stem}.log"), encoding="utf-8", errors="replace").read()
    problems = []
    for pat, msg in [(r"Undefined control sequence", "undefined control sequence"), (r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", "undefined reference"),
                     (r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", "undefined citation"), (r"^! ", "LaTeX error")]:
        hits = re.findall(pat, log, flags=re.M)
        if hits:
            problems.append(f"{msg}: {len(hits)}")
    body = ""
    for f in os.listdir(PAPER):
        if f.endswith(".tex") and f not in ("numbers.tex", "scirep_style.tex"):
            body += open(os.path.join(PAPER, f), encoding="utf-8").read()
    # macros without a value are a problem only when the text uses them (many undefined-by-design quantities,
    # e.g. the naive predictor's within-drug correlation on unseen cells, are emitted as "--" and never cited)
    empty = set(re.findall(r"\\newcommand\{\\(\w+)\}\{--\}", open(os.path.join(PAPER, "numbers.tex"), encoding="utf-8").read()))
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    bad = sorted(empty & used)
    if bad:
        problems.append(f"{len(bad)} referenced macros without a value: {', '.join(bad[:8])}")
    todo = re.findall(r"%\s*TODO[^\n]*", body)
    if todo:
        problems.append(f"{len(todo)} TODO markers in the text")
    print("CHECK " + ("OK" if not problems else "FAIL: " + "; ".join(problems)))
    return not problems


def main():
    py = sys.executable
    if "--tex-only" not in sys.argv:
        if run([py, "-W", "ignore", os.path.join(SRC, "analyze.py")]) != 0:
            sys.exit("analyze.py failed")
        if run([py, "-W", "ignore", os.path.join(SRC, "figs_tables.py")]) != 0:
            sys.exit("figs_tables.py failed")
    if not os.path.exists(os.path.join(PAPER, "main.tex")):
        print("manuscript sources are not distributed with this repository: tables, figures and numbers.tex regenerated; LaTeX step skipped")
        sys.exit(0)
    compile_tex("main")
    ok = check("main")
    print("built", os.path.join(PAPER, "main.pdf"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
