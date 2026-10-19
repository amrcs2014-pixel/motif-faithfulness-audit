"""Build the modelling dataset: GDSC2 responses x DepMap 24Q2 multi-omics x drug SMILES/targets.
Writes data/dataset.npz and data/meta.json.  Usage: python prep_data.py
"""
import os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# GDSC PUTATIVE_TARGET tokens that are not HGNC symbols -> gene symbols
ALIAS = {
    "BCL-XL": ["BCL2L1"], "BCL-W": ["BCL2L2"], "MTORC1": ["MTOR"], "MTORC2": ["MTOR"],
    "PI3K (CLASS 1)": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"], "PI3K (CLASS I)": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"], "PI3KALPHA": ["PIK3CA"], "PI3KBETA": ["PIK3CB"],
    "PI3KDELTA": ["PIK3CD"], "PI3KGAMMA": ["PIK3CG"], "P110ALPHA": ["PIK3CA"], "P110BETA": ["PIK3CB"],
    "P110DELTA": ["PIK3CD"], "P110GAMMA": ["PIK3CG"], "ERK1": ["MAPK3"], "ERK2": ["MAPK1"], "ERK": ["MAPK1", "MAPK3"],
    "MEK1": ["MAP2K1"], "MEK2": ["MAP2K2"], "MEK": ["MAP2K1", "MAP2K2"], "JNK": ["MAPK8", "MAPK9"], "JNK1": ["MAPK8"],
    "P38": ["MAPK14"], "P38 MAPK": ["MAPK14"], "P38ALPHA": ["MAPK14"], "AURORA A": ["AURKA"], "AURORA B": ["AURKB"],
    "AURKA/B": ["AURKA", "AURKB"], "HSP90": ["HSP90AA1", "HSP90AB1"], "TOP2": ["TOP2A"], "TOPOISOMERASE II": ["TOP2A"],
    "ABL": ["ABL1"], "RAF": ["RAF1", "BRAF"], "CRAF": ["RAF1"], "DNAPK": ["PRKDC"], "DNA-PK": ["PRKDC"],
    "TNKS1": ["TNKS"], "PDGFR": ["PDGFRA", "PDGFRB"], "VEGFR": ["KDR", "FLT1"], "VEGFR2": ["KDR"], "VEGFR1": ["FLT1"],
    "VEGFR3": ["FLT4"], "TAK1": ["MAP3K7"], "MPS1": ["TTK"], "KSP": ["KIF11"], "EG5": ["KIF11"], "IKK": ["CHUK", "IKBKB"],
    "IKKB": ["IKBKB"], "IKKA": ["CHUK"], "TUBULIN": ["TUBB", "TUBA1A"], "MICROTUBULE STABILISER": ["TUBB", "TUBA1A"],
    "MICROTUBULE DESTABILISER": ["TUBB", "TUBA1A"], "CDC25": ["CDC25A", "CDC25B"], "BRD2/3/4": ["BRD2", "BRD3", "BRD4"],
    "BET": ["BRD2", "BRD3", "BRD4"], "P53": ["TP53"], "AKT": ["AKT1", "AKT2", "AKT3"], "ROCK": ["ROCK1", "ROCK2"],
    "GSK3": ["GSK3A", "GSK3B"], "HDAC": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"], "HDAC1/2/3/6": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
    "FGFR": ["FGFR1", "FGFR2", "FGFR3"], "JAK": ["JAK1", "JAK2"], "PIM": ["PIM1", "PIM2", "PIM3"], "SRC FAMILY": ["SRC"],
    "SMOOTHENED": ["SMO"], "XIAP": ["XIAP"], "CIAP1": ["BIRC2"], "CIAP2": ["BIRC3"], "HER2": ["ERBB2"], "TRKA": ["NTRK1"],
    "TRKB": ["NTRK2"], "TRKC": ["NTRK3"], "CSF1R": ["CSF1R"], "ATM": ["ATM"], "PARP": ["PARP1", "PARP2"],
    "PKC": ["PRKCA", "PRKCB"], "PKA": ["PRKACA"], "PDK1 (PDPK1)": ["PDPK1"], "PDK1": ["PDPK1"], "S6K1": ["RPS6KB1"],
    "RSK": ["RPS6KA1", "RPS6KA3"], "P90RSK": ["RPS6KA1"], "CHK1": ["CHEK1"], "CHK2": ["CHEK2"], "AMPK": ["PRKAA1"],
    "AKT1/2": ["AKT1", "AKT2"], "G9A": ["EHMT2"], "GLP": ["EHMT1"], "LSD1": ["KDM1A"], "DOT1L": ["DOT1L"],
    "EZH2": ["EZH2"], "NAE": ["NAE1", "UBA3"], "PROTEASOME": ["PSMB5"], "RPS6KB1": ["RPS6KB1"], "RIPK1": ["RIPK1"],
    "TGFBR1": ["TGFBR1"], "ALK5": ["TGFBR1"], "IAP": ["XIAP", "BIRC2", "BIRC3"], "SURVIVIN": ["BIRC5"], "ABL1": ["ABL1"],
    "ERBB2": ["ERBB2"], "KRAS (G12C)": ["KRAS"], "RAS": ["KRAS", "NRAS", "HRAS"], "PLK": ["PLK1"], "CDK": ["CDK1", "CDK2"],
    "CDK1/2": ["CDK1", "CDK2"], "CDK4/6": ["CDK4", "CDK6"], "WEE1": ["WEE1"], "PKMYT1": ["PKMYT1"], "DHFR": ["DHFR"],
    "RRM1": ["RRM1"], "DNA SYNTHESIS": ["TYMS"], "ANTIMETABOLITE": ["TYMS", "DHFR"], "TYMS": ["TYMS"], "TOP1": ["TOP1"],
    "ANDROGEN RECEPTOR": ["AR"], "ESTROGEN RECEPTOR": ["ESR1"], "ER": ["ESR1"], "GR": ["NR3C1"], "RAR": ["RARA"],
    "RXR": ["RXRA"], "PPARG": ["PPARG"], "HIF1A": ["HIF1A"], "BRAF V600E": ["BRAF"], "MDM2": ["MDM2"], "MDMX": ["MDM4"],
    "USP7": ["USP7"], "USP1": ["USP1"], "USP": ["USP7"], "MCL1": ["MCL1"], "BCL2": ["BCL2"], "BCL-2": ["BCL2"],
    "ACTIVATOR OF CASPASE": ["CASP3"], "CASPASE": ["CASP3"], "TTK": ["TTK"], "SIRT1": ["SIRT1"], "SIRT2": ["SIRT2"],
    "BMP": ["BMPR1A"], "EP300": ["EP300"], "CBP": ["CREBBP"], "CREBBP": ["CREBBP"], "BRD9": ["BRD9"], "BRPF1": ["BRPF1"],
    "KDM4": ["KDM4A"], "KDM5": ["KDM5A"], "PRMT5": ["PRMT5"], "FAK": ["PTK2"], "FAK2": ["PTK2B"], "PYK2": ["PTK2B"],
    "LCK": ["LCK"], "LYN": ["LYN"], "FYN": ["FYN"], "YES": ["YES1"], "TIE2": ["TEK"], "AXL": ["AXL"], "MERTK": ["MERTK"],
    "IGF1R": ["IGF1R"], "IR": ["INSR"], "INSR": ["INSR"], "EPHB4": ["EPHB4"], "DDR1": ["DDR1"], "DDR2": ["DDR2"],
    "ITK": ["ITK"], "BLK": ["BLK"], "TEC": ["TEC"], "BMX": ["BMX"], "GSK3B": ["GSK3B"], "CK2": ["CSNK2A1"],
    "CSNK1D": ["CSNK1D"], "CSNK1E": ["CSNK1E"], "CK1": ["CSNK1D", "CSNK1E"], "PAK": ["PAK1"], "MST1": ["STK4"], "MST2": ["STK3"],
    "LRRK2": ["LRRK2"], "NEK2": ["NEK2"], "NUAK1": ["NUAK1"], "VPS34": ["PIK3C3"], "PIK3C3": ["PIK3C3"],
    "MNK1": ["MKNK1"], "MNK2": ["MKNK2"], "MNK": ["MKNK1", "MKNK2"], "MAP4K4": ["MAP4K4"], "TBK1": ["TBK1"],
    "PERK": ["EIF2AK3"], "GCN2": ["EIF2AK4"], "PKR": ["EIF2AK2"], "SGK": ["SGK1"], "SGK3": ["SGK3"], "ULK1": ["ULK1"],
    "TAF1": ["TAF1"], "ATAD2": ["ATAD2"], "DYRK1A": ["DYRK1A"], "CLK": ["CLK1"], "HIPK": ["HIPK2"], "PI4K": ["PI4KA"],
    "PIK3CA": ["PIK3CA"], "MTOR": ["MTOR"], "EGFR": ["EGFR"], "RET": ["RET"], "KIT": ["KIT"], "MET": ["MET"],
    "ALK": ["ALK"], "ROS1": ["ROS1"], "FLT3": ["FLT3"], "BTK": ["BTK"], "SYK": ["SYK"], "IRAK4": ["IRAK4"], "IRAK1": ["IRAK1"],
    "GAMMA-SECRETASE": ["PSEN1"], "NOTCH": ["NOTCH1"], "WNT": ["PORCN"], "PORCN": ["PORCN"], "TANKYRASE": ["TNKS", "TNKS2"],
    "XPO1": ["XPO1"], "CRM1": ["XPO1"], "NAMPT": ["NAMPT"], "IDH1": ["IDH1"], "IDH2": ["IDH2"], "DNMT1": ["DNMT1"],
    "HSP70": ["HSPA1A"], "HSF1": ["HSF1"], "VCP": ["VCP"], "ATR": ["ATR"], "FEN1": ["FEN1"], "APE1": ["APEX1"],
    "TOPOISOMERASE I": ["TOP1"], "TOPO1": ["TOP1"], "TOPO2": ["TOP2A"], "AKT1": ["AKT1"], "SRC": ["SRC"], "BRAF": ["BRAF"],
}


def sym(col):
    m = re.match(r"^(.+?) \(\d+\)$", col)
    return m.group(1) if m else col


def parse_targets(s, symbols):
    if not isinstance(s, str): return []
    out = []
    for tok in re.split(r"[,;/]", s):
        t = tok.strip()
        if not t or t.lower() in ("not defined", "others", "other", "unknown"): continue
        T = t.upper()
        if T in ALIAS:
            out += ALIAS[T]
        elif t in symbols:
            out.append(t)
        elif T in symbols:
            out.append(T)
        else:
            # tokens like "CDK4, CDK6" already split; try removing spaces/parentheses
            T2 = re.sub(r"\s*\(.*?\)", "", T).strip()
            if T2 in ALIAS: out += ALIAS[T2]
            elif T2 in symbols: out.append(T2)
    return sorted(set(g for g in out if g in symbols))


def main():
    print("loading GDSC2 ...", flush=True)
    resp = pd.read_csv(os.path.join(RAW, "GDSC2_fitted_dose_response_27Oct23.csv"))
    resp = resp[resp.DATASET == "GDSC2"]
    smi = pd.read_csv(os.path.join(DATA, "drug_smiles.csv"))
    smi = smi[smi.smiles.notna() & (smi.smiles.astype(str) != "")]
    from rdkit import Chem
    ok = {}
    for r in smi.itertuples():
        m = Chem.MolFromSmiles(str(r.smiles))
        if m is None or m.GetNumHeavyAtoms() < 3: continue
        # keep largest organic fragment (salts), require carbon
        frags = Chem.GetMolFrags(m, asMols=True)
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        if not any(a.GetSymbol() == "C" for a in m.GetAtoms()): continue
        ok[int(r.drug_id)] = Chem.MolToSmiles(m)
    print(f"drugs with parsable organic SMILES: {len(ok)}/{resp.DRUG_ID.nunique()}")
    # some drug names occur under two DRUG_IDs (different screening sites): keep both ids as distinct drugs
    resp = resp[resp.DRUG_ID.isin(ok)]

    model = pd.read_csv(os.path.join(DEPMAP, "Model.csv"))
    s2m = dict(zip(model.SangerModelID.dropna(), model.ModelID[model.SangerModelID.notna()]))
    resp["ModelID"] = resp.SANGER_MODEL_ID.map(s2m)
    print("cells mapped to DepMap:", resp.ModelID.notna().sum(), "/", len(resp), "rows;",
          resp.loc[resp.ModelID.notna(), "SANGER_MODEL_ID"].nunique(), "cells")
    resp = resp[resp.ModelID.notna()]

    print("loading omics ...", flush=True)
    expr = pd.read_csv(os.path.join(DEPMAP, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"), index_col=0)
    expr.columns = [sym(c) for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    dmg = pd.read_csv(os.path.join(DEPMAP, "OmicsSomaticMutationsMatrixDamaging.csv"), index_col=0)
    dmg.columns = [sym(c) for c in dmg.columns]; dmg = dmg.loc[:, ~dmg.columns.duplicated()]
    hot = pd.read_csv(os.path.join(DEPMAP, "OmicsSomaticMutationsMatrixHotspot.csv"), index_col=0)
    hot.columns = [sym(c) for c in hot.columns]; hot = hot.loc[:, ~hot.columns.duplicated()]
    cnv = pd.read_csv(os.path.join(DEPMAP, "OmicsCNGene.csv"), index_col=0)
    cnv.columns = [sym(c) for c in cnv.columns]; cnv = cnv.loc[:, ~cnv.columns.duplicated()]
    crispr = pd.read_csv(os.path.join(DEPMAP, "CRISPRGeneEffect.csv"), index_col=0)
    crispr.columns = [sym(c) for c in crispr.columns]; crispr = crispr.loc[:, ~crispr.columns.duplicated()]

    cells = sorted(set(resp.ModelID) & set(expr.index) & set(dmg.index) & set(cnv.index))
    print("cells with expression+mutation+CNV:", len(cells))
    resp = resp[resp.ModelID.isin(cells)]

    # gene panel: GDSC putative targets (mapped) + top-variance expression genes
    symbols = set(expr.columns)
    tg = {}
    for r in resp[["DRUG_ID", "PUTATIVE_TARGET"]].drop_duplicates().itertuples():
        tg[int(r.DRUG_ID)] = parse_targets(r.PUTATIVE_TARGET, symbols)
    target_genes = sorted(set(g for v in tg.values() for g in v))
    print("drugs with >=1 mapped target gene:", sum(1 for v in tg.values() if v), "/", len(tg), "; target genes:", len(target_genes))
    unmapped = sorted(set(str(t) for d, t in resp[["DRUG_ID", "PUTATIVE_TARGET"]].drop_duplicates().itertuples(index=False) if not tg[int(d)]))
    print("unmapped target strings:", unmapped[:60])
    E = expr.loc[cells]
    var = E.var(axis=0).sort_values(ascending=False)
    topvar = [g for g in var.index[:N_TOPVAR_GENES]]
    genes = sorted(set(target_genes) | set(topvar))
    print("gene panel:", len(genes))

    X_expr = E[genes].to_numpy(np.float32)
    M = pd.DataFrame(0.0, index=cells, columns=genes)
    common = [g for g in genes if g in dmg.columns]
    M.loc[:, common] = dmg.loc[cells, common].to_numpy()
    commonh = [g for g in genes if g in hot.columns]
    M.loc[:, commonh] = np.maximum(M.loc[:, commonh].to_numpy(), hot.reindex(index=cells, columns=commonh).fillna(0).to_numpy())
    X_mut = (M.to_numpy(np.float32) > 0).astype(np.float32)
    C = cnv.reindex(index=cells, columns=genes)
    C = C.fillna(C.median())  # few missing genes -> median
    X_cnv = C.to_numpy(np.float32)
    K = crispr.reindex(index=cells, columns=genes).to_numpy(np.float32)  # NaN where no CRISPR screen
    print("cells with CRISPR:", int((~np.isnan(K).all(axis=1)).sum()))

    # drugs and pairs
    drug_ids = sorted(resp.DRUG_ID.unique())
    d_index = {d: i for i, d in enumerate(drug_ids)}
    c_index = {c: i for i, c in enumerate(cells)}
    dn = resp.groupby("DRUG_ID").DRUG_NAME.first()
    dp = resp.groupby("DRUG_ID").PATHWAY_NAME.first()
    dt = resp.groupby("DRUG_ID").PUTATIVE_TARGET.first()
    # average replicate measurements of the same (drug, cell)
    agg = resp.groupby(["DRUG_ID", "ModelID"]).agg(LN_IC50=("LN_IC50", "mean"), AUC=("AUC", "mean")).reset_index()
    pairs_d = agg.DRUG_ID.map(d_index).to_numpy(np.int32)
    pairs_c = agg.ModelID.map(c_index).to_numpy(np.int32)
    y = agg.LN_IC50.to_numpy(np.float32); auc = agg.AUC.to_numpy(np.float32)
    print("pairs:", len(y), "drugs:", len(drug_ids), "cells:", len(cells))

    lineage = model.set_index("ModelID").OncotreeLineage.reindex(cells).fillna("Unknown").tolist()
    sanger = model.set_index("ModelID").SangerModelID.reindex(cells).tolist()
    cname = model.set_index("ModelID").CellLineName.reindex(cells).fillna("").tolist()

    np.savez_compressed(os.path.join(DATA, "dataset.npz"), expr=X_expr, mut=X_mut, cnv=X_cnv, crispr=K,
                        pair_drug=pairs_d, pair_cell=pairs_c, y=y, auc=auc)
    meta = {
        "genes": genes, "target_genes": target_genes, "cells": cells, "cell_sanger": sanger, "cell_name": cname,
        "cell_lineage": lineage,
        "drugs": [{"drug_id": int(d), "name": str(dn[d]), "smiles": ok[int(d)], "pathway": str(dp[d]),
                   "target_str": str(dt[d]), "targets": tg.get(int(d), [])} for d in drug_ids],
        "n_pairs": int(len(y)),
    }
    jdump(meta, os.path.join(DATA, "meta.json"))
    print("saved dataset.npz / meta.json")


if __name__ == "__main__":
    main()
