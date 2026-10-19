"""Word counts of the abstract and the main text after macro expansion. Usage: python wordcount.py"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PAPER

macs = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", open(os.path.join(PAPER, "numbers.tex"), encoding="utf-8").read(), flags=re.M))


def expand(t):
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", t)
    t = re.sub(r"\\(?:input|includegraphics|label|ref|cite)\{[^}]*\}", "", t)
    t = re.sub(r"\\(\w+)\{\}", lambda m: macs.get(m.group(1), m.group(0)), t)
    t = re.sub(r"\\(\w+)", lambda m: macs.get(m.group(1), " "), t)
    t = re.sub(r"[\$\{\}\[\]]", "", t)
    return t


for f in ("abstract.tex", "intro.tex", "results.tex", "discussion.tex", "methods.tex"):
    t = expand(open(os.path.join(PAPER, f), encoding="utf-8").read())
    print(f"{f:16s} {len(t.split()):6d} words")
