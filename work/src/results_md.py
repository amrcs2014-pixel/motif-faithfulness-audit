"""Write ../../RESULTS.md (project root) and fill the key-results block of README_final.md from the built numbers.
Usage: python results_md.py   (after build.py)
"""
import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

PROJECT = os.path.dirname(ROOT)
macs = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", open(os.path.join(PAPER, "numbers.tex"), encoding="utf-8").read(), flags=re.M))


def m(k):
    v = macs.get(k, "--")
    return v.replace("\\%", "%").replace("{,}", ",").replace("$", "").replace("\\rho", "rho").replace("\\,", " ").replace("\\pm", "+/-")


acc = pd.read_csv(os.path.join(SUMMARY, "accuracy.csv"))
acc["key"] = acc.apply(lambda r: f"{r.model}|{r.vocab}" if r.model == "ccmg" else r.model, axis=1)
LAB = {"naive": "naive mean effect", "lgbm": "LightGBM (tuned)", "latefusion": "late fusion", "ccmg|brics": "MotifCell-GNN (BRICS)", "ccmg|recap": "MotifCell-GNN (RECAP)"}
SL = {"random": "random", "unseen_drug": "unseen drug", "unseen_cell": "unseen cell"}
rows = ["| Model | " + " | ".join(f"{SL[s]} RMSE | {SL[s]} r_drug" for s in SPLITS) + " |", "|---|" + "---|" * 6]
for k in ["naive", "lgbm", "latefusion", "ccmg|brics", "ccmg|recap"]:
    cells = []
    for s in SPLITS:
        g = acc[(acc.key == k) & (acc.split == s)]
        if len(g):
            g = g.iloc[0]
            cells += [f"{g.test_rmse:.3f}", "--" if pd.isna(g.test_per_drug_pearson) else f"{g.test_per_drug_pearson:.3f}"]
        else:
            cells += ["--", "--"]
    rows.append(f"| {LAB[k]} | " + " | ".join(cells) + " |")
acc_table = "\n".join(rows)

faith = pd.read_csv(os.path.join(SUMMARY, "faithfulness_agg.csv"))
fa = faith[(faith.counterfactual == "delta_del")].groupby(["vocab", "explainer"]).rho.mean().unstack()
order = ["attn", "rollout", "ig", "mask", "gnnexp", "gnnexp2", "size", "random"]
FL = {"attn": "attention", "rollout": "rollout", "ig": "integrated gradients", "mask": "masking", "gnnexp": "GNNExplainer", "gnnexp2": "GNNExplainer (input mask)", "size": "motif size", "random": "random"}
frows = ["| Explainer | BRICS rho | RECAP rho |", "|---|---|---|"]
for ex in order:
    if ex in fa.columns:
        frows.append(f"| {FL[ex]} | {fa.loc['brics', ex]:.2f} | {fa.loc['recap', ex]:.2f} |" if "recap" in fa.index else f"| {FL[ex]} | {fa.loc['brics', ex]:.2f} | -- |")
faith_table = "\n".join(frows)

key = f"""
**Accuracy (fixed split, 5 seeds; within-drug Pearson r_drug is the DrEval-style metric)**

{acc_table}

**Stability across training seeds (random regime, BRICS, 10 seed pairs; Spearman rho between the two explanations of the same probe pair)**
- predictions: rho = {m('stabSeedRandomMPredictionRho')}
- attention-ranked motifs: rho = {m('stabSeedRandomMAttnRho')} (top-1 agreement {m('stabSeedRandomMAttnTop')}, chance {m('faithBricsChanceTop')}); masking {m('stabSeedRandomMMaskRho')}; integrated gradients {m('stabSeedRandomMIgRho')}; rollout {m('stabSeedRandomMRolloutRho')}
- the model's own deletion effects Delta_del: rho = {m('stabSeedRandomMDdelRho')}
- gene rankings (pooled over regimes): rho {m('stabSeedGeneRhoMin')}-{m('stabSeedGeneRhoMax')}, top-20 Jaccard {m('stabSeedGeneJacMin')}-{m('stabSeedGeneJacMax')}
- input-noise stability (sigma = {m('perturbSigma')}): every explainer rho >= {m('stabPerturbMGnnexpRho')}; prediction {m('stabPerturbMPredictionRho')}

**Cell conditioning**: attention names the same top motif in {m('cdBricsAttnModal')} of a drug's probe cells (RECAP {m('cdRecapAttnModal')}); largest attention weight {m('attnMaxBrics')} on average, > 0.9 in {m('attnPeakedBrics')} of pairs; the deletion effect itself names the same motif in {m('cdBricsDdelModal')} of cells.

**Counterfactual faithfulness (mean per-pair Spearman between |score| and |Delta_del|, pooled over regimes and seeds)**

{faith_table}

Top-1 agreement with the largest deletion effect (BRICS): attention {m('faithBricsDelAttnTop')}, rollout {m('faithBricsDelRolloutTop')}, IG {m('faithBricsDelIgTop')}, masking {m('faithBricsDelMaskTop')}, GNNExplainer {m('faithBricsDelGnnexpTop')} (chance {m('faithBricsChanceTop')}). Valid-edit vs graph-edit deletions give the same picture (attention {m('faithBricsDelAttnRhoValid')} vs {m('faithBricsDelAttnRhoGraph')}); {m('fracChemValid')} of deletions sanitise in RDKit.

**Fragmentation rule**: for the same pair, seed and regime, the top BRICS and RECAP motifs share no atom in {m('vocabAtomDisjointAttn')} of pairs (attention; atom Jaccard {m('vocabAtomJacAttn')}); deletion effect {m('vocabAtomDisjointDdel')}. Gene-level explanations of the two backbones agree at rho {m('stabVocabGIgRho')} (IG).

**Biology**: an annotated GDSC target is among the top-20 genes in {m('bioBricsGIgHit')} of probe pairs for IG (chance {m('bioBricsGIgChance')}); knockdown counterfactual {m('bioBricsGKdHit')}; dependency agreement rho_dep = {m('bioBricsGIgDep')} (IG) over {m('bioDepTriples')} drug-target-run triples; top-1 motif pathway-enriched in {m('bioBricsMAttnHit')} (attention) vs chance {m('bioBricsMAttnChance')}; cell-conditioning coherence rho = {m('cohRhoAbsDel')} (prediction {m('cohRhoPred')}, measured response {m('cohRhoTrue')}).

**Drug-level seed agreement (random regime, BRICS)**: the modal top motif of a probe drug is identical in all five seeds for {m('dsaBricsAttnAll')} of the 30 probe drugs for attention (pairwise {m('dsaBricsAttnPair')}), {m('dsaBricsMaskAll')} for masking, {m('dsaBricsDdelAll')} for the deletion effect, {m('dsaBricsRandomAll')} for random scores.

**Positive controls**: of {m('nBiomarkerPairs')} drug-biomarker pairs with an established direction, the measured response has the expected sign for {m('bioPcTrueSignOk')} and the prediction for {m('bioPcPredSignOk')}; IG attributions rank TP53 first for nutlin-3a (median rank fraction {m('bioPcNutlinTpRank')}, top-20 in {m('bioPcNutlinTpTop')} of cells) and PIK3CA high for afuresertib ({m('bioPcAfuPikRank')}) but not BRAF for selumetinib ({m('bioPcSelBrafRank')}) nor the expression biomarkers ({m('bioPcExprRankMin')}-{m('bioPcExprRankMax')}). Curated pharmacophores (high confidence, BRICS): top-1 motif is a pharmacophore motif for attention {m('pharmBricsHighAttn')}, IG {m('pharmBricsHighIg')}, masking {m('pharmBricsHighMask')}, deletion effect {m('pharmBricsHighDdel')} vs chance {m('pharmBricsHighChance')} (motif size alone {m('pharmBricsHighSize')}).

**Planted-importance sanity check**: on synthetic labels depending on one motif, attention / rollout / IG / masking / input-mask GNNExplainer recover it in {m('sanityAttn')} / {m('sanityRollout')} / {m('sanityIg')} / {m('sanityMask')} / {m('sanityGnnexpB')} of pairs; the attention-gated GNNExplainer-style mask in {m('sanityGnnexp')} (chance {m('sanityChance')}).

**Controls (random regime, BRICS, 5 seeds each; default -> 4 heads / entropy bonus / tuned {m('tunedTag')})**: within-drug r {m('varDefRd')} -> {m('varHfourRd')} / {m('varEntRd')} / {m('varTunedRd')}; largest attention weight {m('varDefAttnMax')} -> {m('varHfourAttnMax')} / {m('varEntAttnMax')} / {m('varTunedAttnMax')}; same top motif across a drug's cells {m('varDefCdAttnPct')} -> {m('varHfourCdAttnPct')} / {m('varEntCdAttnPct')} / {m('varTunedCdAttnPct')}; seed rho of attention {m('varDefSeedAttn')} -> {m('varHfourSeedAttn')} / {m('varEntSeedAttn')} / {m('varTunedSeedAttn')}; drug-level agreement {m('varDefDrugAttn')} -> {m('varHfourDrugAttn')} / {m('varEntDrugAttn')} / {m('varTunedDrugAttn')}; faithfulness of attention {m('varDefFaithAttn')} -> {m('varHfourFaithAttn')} / {m('varEntFaithAttn')} / {m('varTunedFaithAttn')}. Tuning grid: {m('nTuningConfigs')} configurations, validation RMSE {m('tuningBestVal')}-{m('tuningWorstVal')} (default {m('tuningDefaultVal')}). Seed ensemble r_drug: {m('accCcmgEnsBricsRandomRd')} / {m('accCcmgEnsBricsUnseenDrugRd')} / {m('accCcmgEnsBricsUnseenCellRd')} (random / unseen drug / unseen cell).
"""

doc = f"""# RESULTS — Are cell-conditioned motif explanations faithful?

Build of 16–17 Oct 2026 (`work/`, one GTX 1060). Paper: `Final/Paper_MotifFaithfulness/MotifFaithfulness.pdf`. Every number below is a
macro in `work/paper/numbers.tex`, regenerated by `python work/build.py`.

## Setup in one paragraph
GDSC2 (release 8.5) x DepMap 24Q2: {m('nPairs')} drug–cell pairs, {m('nDrugs')} drugs with organic SMILES, {m('nCells')} cell lines with
expression + mutation + copy number ({m('nCellsCrispr')} with CRISPR), gene panel {m('nGenes')} ({m('nTargetGenes')} mapped GDSC targets + top-1000 variance).
CCMG-class backbone "MotifCell-GNN" (GINE atoms -> BRICS or RECAP motif nodes -> cell node with motif–cell attention; {m('nParams')} parameters),
late-fusion baseline, DrEval naive mean-effect predictor, tuned LightGBM. Regimes random / unseen drug / unseen cell; fixed split x 5
training seeds (+ 4 re-drawn splits for BRICS). Probe set {m('nProbeDrugs')} drugs x {m('nProbeCells')} cells = {m('nProbePairs')} pairs never trained on.
Explainers: attention, rollout, integrated gradients, masking, GNNExplainer-style mask (+ input-mask variant) at motif level; IG,
gradient x input, masking, learned mask at gene level. Counterfactuals: motif deletion (re-featurised when the remainder sanitises),
size-matched motif swap, gene expression knockdown. {m('nTrainRuns')} training runs, {m('nExplainRuns')} explanation runs.

## Key results
{key}

## Verdict
An honest negative for motif-level interpretability claims: explanations are reproducible under input noise but not under re-training
(while predictions are), the model's own attention is a drug-level label rather than a cell-conditioned explanation, attention and a
GNNExplainer-style mask are weakly or not at all faithful to the model's counterfactual behaviour (rollout and integrated gradients are
moderately faithful; masking is faithful by construction), the fragmentation rule changes which atoms are blamed, and agreement with
drug–target biology is above chance but small. The backbone itself beats the naive predictor only under the random regime.

## Files
- `work/results/runs/*/metrics.json` — accuracy per run; `work/results/baselines/` — naive and LightGBM
- `work/results/explain/*/scores.npz` — explanation score matrices and counterfactual effects per run
- `work/results/summary/*.csv` — aggregated tables (accuracy, stability, faithfulness, cell dependence, biology)
- `work/paper/` — LaTeX sources, `numbers.tex`, tables and figures; `Final/` — deliverable package
"""
open(os.path.join(PROJECT, "RESULTS.md"), "w", encoding="utf-8").write(doc)
rf = os.path.join(ROOT, "README_final.md")
t = open(rf, encoding="utf-8").read()
if "__RESULTS_PLACEHOLDER__" in t:
    t = t.replace("__RESULTS_PLACEHOLDER__", key.strip())
else:  # refresh the block between the key-results heading and the rebuilding section
    a = t.index("## Key results"); b = t.index("## Rebuilding")
    head = t[:a]; tail = t[b:]
    t = head + "## Key results (see RESULTS.md in the project root for the full summary)\n\n" + key.strip() + "\n\n" + tail
open(rf, "w", encoding="utf-8").write(t)
print("wrote RESULTS.md and README_final.md key results")
