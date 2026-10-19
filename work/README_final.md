# Are cell-conditioned motif explanations faithful? — deliverable package

Paper: *Are cell-conditioned motif explanations faithful? A stability and counterfactual audit of motif-level
interpretability in multi-omics drug response prediction* (Scientific Reports layout, single author).

## Contents

| Path | What it is |
|---|---|
| `Paper_MotifFaithfulness/MotifFaithfulness.pdf` | compiled manuscript |
| `Paper_MotifFaithfulness/latex_source/` | standalone LaTeX sources (main.tex + section files, numbers.tex, tables, figures, refs.bib, naturemag.bst); compiles with `pdflatex` + `bibtex` |
| `Paper_MotifFaithfulness/latex_source.zip` | the same sources zipped for submission |
| `Paper_MotifFaithfulness/cover_letter.md` | cover letter draft |
| `README.md` | this file |

The reproducible pipeline is in `../work/` (not copied here; see "Rebuilding").

## Key results (see RESULTS.md in the project root for the full summary)

**Accuracy (fixed split, 5 seeds; within-drug Pearson r_drug is the DrEval-style metric)**

| Model | random RMSE | random r_drug | unseen drug RMSE | unseen drug r_drug | unseen cell RMSE | unseen cell r_drug |
|---|---|---|---|---|---|---|
| naive mean effect | 1.199 | 0.629 | 2.269 | 0.648 | 1.423 | -- |
| LightGBM (tuned) | 1.049 | 0.715 | 2.444 | 0.657 | 1.250 | 0.453 |
| late fusion | 1.033 | 0.725 | 2.081 | 0.642 | 1.304 | 0.411 |
| MotifCell-GNN (BRICS) | 1.007 | 0.743 | 2.844 | 0.616 | 1.297 | 0.435 |
| MotifCell-GNN (RECAP) | 1.022 | 0.732 | 2.760 | 0.584 | 1.297 | 0.432 |

**Stability across training seeds (random regime, BRICS, 10 seed pairs; Spearman rho between the two explanations of the same probe pair)**
- predictions: rho = 0.96
- attention-ranked motifs: rho = 0.36 (top-1 agreement 0.43, chance 0.20); masking 0.42; integrated gradients 0.37; rollout 0.36
- the model's own deletion effects Delta_del: rho = 0.40
- gene rankings (pooled over regimes): rho 0.25-0.54, top-20 Jaccard 0.03-0.32
- input-noise stability (sigma = 0.25): every explainer rho >= 0.85; prediction 1.00

**Cell conditioning**: attention names the same top motif in 91% of a drug's probe cells (RECAP 92%); largest attention weight 0.85 on average, > 0.9 in 59% of pairs; the deletion effect itself names the same motif in 76% of cells.

**Counterfactual faithfulness (mean per-pair Spearman between |score| and |Delta_del|, pooled over regimes and seeds)**

| Explainer | BRICS rho | RECAP rho |
|---|---|---|
| attention | 0.22 | 0.18 |
| rollout | 0.49 | 0.35 |
| integrated gradients | 0.40 | 0.31 |
| masking | 0.93 | 0.83 |
| GNNExplainer | -0.01 | 0.10 |
| GNNExplainer (input mask) | 0.43 | 0.34 |
| motif size | 0.12 | -0.08 |
| random | -0.00 | 0.00 |

Top-1 agreement with the largest deletion effect (BRICS): attention 0.48, rollout 0.50, IG 0.43, masking 0.95, GNNExplainer 0.12 (chance 0.20). Valid-edit vs graph-edit deletions give the same picture (attention 0.24 vs 0.32); 85% of deletions sanitise in RDKit.

**Fragmentation rule**: for the same pair, seed and regime, the top BRICS and RECAP motifs share no atom in 66% of pairs (attention; atom Jaccard 0.26); deletion effect 64%. Gene-level explanations of the two backbones agree at rho 0.55 (IG).

**Biology**: an annotated GDSC target is among the top-20 genes in 6.0% of probe pairs for IG (chance 3.6%); knockdown counterfactual 5.6%; dependency agreement rho_dep = 0.02 (IG) over 1,674 drug-target-run triples; top-1 motif pathway-enriched in 46% (attention) vs chance 39%; cell-conditioning coherence rho = 0.01 (prediction -0.01, measured response -0.06).

**Drug-level seed agreement (random regime, BRICS)**: the modal top motif of a probe drug is identical in all five seeds for 10% of the 30 probe drugs for attention (pairwise 46%), 20% for masking, 20% for the deletion effect, 0% for random scores.

**Positive controls**: of 12 drug-biomarker pairs with an established direction, the measured response has the expected sign for 8 and the prediction for 8; IG attributions rank TP53 first for nutlin-3a (median rank fraction 0.003, top-20 in 60% of cells) and PIK3CA high for afuresertib (0.170) but not BRAF for selumetinib (0.583) nor the expression biomarkers (0.34-0.57). Curated pharmacophores (high confidence, BRICS): top-1 motif is a pharmacophore motif for attention 32%, IG 31%, masking 32%, deletion effect 32% vs chance 29% (motif size alone 70%).

**Planted-importance sanity check**: on synthetic labels depending on one motif, attention / rollout / IG / masking / input-mask GNNExplainer recover it in 100% / 100% / 100% / 100% / 100% of pairs; the attention-gated GNNExplainer-style mask in 20% (chance 18%).

**Controls (random regime, BRICS, 5 seeds each; default -> 4 heads / entropy bonus / tuned learning rate 3\times10^{-4})**: within-drug r 0.74 -> 0.74 / 0.75 / 0.76; largest attention weight 0.90 -> 0.59 / 0.58 / 0.74; same top motif across a drug's cells 92% -> 86% / 71% / 84%; seed rho of attention 0.36 -> 0.37 / 0.31 / 0.38; drug-level agreement 46% -> 50% / 40% / 48%; faithfulness of attention 0.21 -> 0.29 / 0.22 / 0.24. Tuning grid: 12 configurations, validation RMSE 0.967-1.194 (default 1.003). Seed ensemble r_drug: 0.771 / 0.667 / 0.456 (random / unseen drug / unseen cell).

## Rebuilding every number

```
cd work
python src/fetch_smiles.py      # PubChem SMILES for the GDSC2 compounds (cached in data/drug_smiles.csv)
python src/prep_data.py         # GDSC2 x DepMap dataset -> data/dataset.npz, data/meta.json
python src/chem.py              # BRICS / RECAP motif graphs -> data/graphs_{brics,recap}.pkl
python src/splits.py            # probe set and the three exposure regimes -> data/splits.npz
python src/gpu_queue.py         # all training (train.py) and explanation (explain.py) jobs, one GPU job at a time
python src/cpu_jobs.py          # naive mean-effect and tuned LightGBM baselines (baselines.py)
python build.py                 # analyze.py -> figs_tables.py -> pdflatex; prints CHECK OK when no macro is missing
python package.py               # assembles this Final/ folder
```

The two queues are resumable (a job is skipped when its result file exists). On the author's laptop they were run as
self-restarting Windows scheduled tasks (`new1_gpu`, `new1_cpu`) with `PYTHONPATH=work\pylib`.

Raw inputs (not included, public): GDSC2 fitted dose-response release 8.5 (27 Oct 2023) and the screened-compound table
from cancerrxgene.org; DepMap 24Q2 `Model.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`,
`OmicsSomaticMutationsMatrixDamaging.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, `OmicsCNGene.csv`,
`CRISPRGeneEffect.csv` from depmap.org; PubChem PUG-REST for structures.

Software: Python 3.11, PyTorch 2.14 (CUDA 12.6), PyTorch Geometric 2.8, RDKit 2026.09, LightGBM 4.7, scikit-learn 1.6,
SciPy, pandas, matplotlib; MiKTeX for the PDF. Hardware: one NVIDIA GeForce GTX 1060 (6 GB).
