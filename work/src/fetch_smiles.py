"""Resolve SMILES for every GDSC2 drug from PubChem (manual alias, name, synonyms), appending each result to
data/drug_smiles.csv as soon as it is known, so the job can be killed and resumed.
Usage: python fetch_smiles.py
"""
import csv, json, os, sys, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "drug_smiles.csv")
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/SMILES,ConnectivitySMILES/JSON"
PUG_OLD = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES,CanonicalSMILES/JSON"
FIELDS = ["drug_id", "drug_name", "query", "cid", "smiles"]

# GDSC names PubChem does not know under that spelling
MANUAL = {
    "Nutlin-3a (-)": "Nutlin-3a", "Sapitinib": "AZD8931", "AZ960": "AZ-960", "WZ4003": "WZ-4003",
    "ML323": "ML-323", "XAV939": "XAV-939", "BIBR-1532": "BIBR 1532", "PD173074": "PD-173074",
    "GSK591": "GSK-591", "LJI308": "LJI-308", "GSK2606414": "GSK-2606414", "AZD5438": "AZD-5438",
    "AZD6482": "AZD-6482", "AZD6738": "Ceralasertib", "AZD7762": "AZD-7762", "AZD8055": "AZD-8055",
    "AZD1332": "AZD-1332", "AZD1208": "AZD-1208", "AZD3759": "Zorifertinib", "AZD4547": "AZD-4547",
    "AZD5363": "Capivasertib", "AZD5582": "AZD-5582", "AZD5991": "AZD-5991", "AZD2014": "Vistusertib",
    "AZD7648": "AZD-7648", "AZD6094": "Savolitinib", "AZD5153": "AZD-5153", "AZD8186": "AZD-8186",
    "JQ1": "(+)-JQ1", "VE821": "VE-821", "VE-822": "Berzosertib", "OTX015": "Birabresib",
    "Picolinici-acid": "Picolinic acid", "PFI3": "PFI-3", "P22077": "P-22077", "SB505124": "SB-505124",
    "SB216763": "SB-216763", "WIKI4": "WIKI-4", "AGK2": "AGK-2", "AT13148": "AT-13148", "ABT737": "ABT-737",
    "AZ6102": "AZ-6102", "LCL161": "LCL-161", "GSK2578215A": "GSK-2578215A", "GSK269962A": "GSK-269962A",
    "PRIMA-1MET": "APR-246", "SCH772984": "SCH-772984", "Wee1 Inhibitor": "PD0166285",
    "EPZ004777": "EPZ-004777", "EPZ5676": "Pinometostat", "KRAS (G12C) Inhibitor-12": "KRAS G12C inhibitor 12",
    "MK-1775": "Adavosertib", "PLX-4720": "PLX4720", "RVX-208": "Apabetalone", "SGC0946": "SGC-0946",
    "TW 37": "TW-37", "Zoledronate": "Zoledronic acid", "ZM447439": "ZM-447439", "BI-2536": "BI 2536",
    "CZC24832": "CZC-24832", "AZD3514": "AZD-3514", "AZD1480": "AZD-1480", "SB590885": "SB-590885",
    "GSK1904529A": "GSK-1904529A", "NU7441": "NU-7441", "PD0325901": "PD-0325901", "PI-103": "PI-103",
    "LGK974": "WNT974", "AZD4877": "AZD-4877", "Nutlin-3a": "Nutlin-3a", "GSK2110183B": "GSK2110183B",
    "GSK2256098C": "GSK2256098", "GSK2276186C": "GSK2276186", "GSK626616AC": "GSK626616",
    "GSK3337463A": "GSK3337463", "GSK2830371A": "GSK2830371", "AZD5582": "AZD-5582", "Avagacestat": "Avagacestat",
    "Ulixertinib": "Ulixertinib", "Entinostat": "Entinostat", "Alpelisib": "Alpelisib", "Pevonedistat": "Pevonedistat",
    "PCI-34051": "PCI-34051", "UMI-77": "UMI-77", "WEHI-539": "WEHI-539", "MIM1": "MIM1", "A-366": "A-366",
    "AMG-319": "AMG-319", "OSI-027": "OSI-027", "PF-562271": "PF-562271", "GNE-317": "GNE-317", "I-BRD9": "I-BRD9",
    "BDP-00009066": "BDP-00009066", "BPD-00008900": "BPD-00008900", "MN-64": "MN-64", "Dihydrorotenone": "Dihydrorotenone",
    "NVP-ADW742": "NVP-ADW742", "PRT062607": "PRT062607", "Sabutoclax": "Sabutoclax", "KU-55933": "KU-55933",
    "MK-2206": "MK-2206", "MK-8776": "MK-8776", "PF-4708671": "PF-4708671", "BMS-754807": "BMS-754807",
    "Ibrutinib": "Ibrutinib", "Piperlongumine": "Piperlongumine", "Elephantin": "Elephantin", "Taselisib": "Taselisib",
}


def fetch(name):
    for url in (PUG, PUG_OLD):
        u = url.format(urllib.parse.quote(name, safe=""))
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                js = json.load(r)
            p = js["PropertyTable"]["Properties"][0]
            smi = p.get("SMILES") or p.get("IsomericSMILES") or p.get("ConnectivitySMILES") or p.get("CanonicalSMILES")
            if smi:
                return smi, p.get("CID")
        except Exception:
            time.sleep(0.3)
    return None, None


def main():
    import pandas as pd
    resp = pd.read_csv(os.path.join(RAW, "GDSC2_fitted_dose_response_27Oct23.csv"), usecols=["DRUG_ID", "DRUG_NAME"]).drop_duplicates()
    comp = pd.read_csv(os.path.join(RAW, "screened_compounds_rel_8.5.csv"))
    syn = {}
    for r in comp.itertuples():
        syn.setdefault(int(r.DRUG_ID), [])
        if isinstance(r.SYNONYMS, str):
            syn[int(r.DRUG_ID)] += [s.strip() for s in r.SYNONYMS.split(",") if s.strip()]
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(int(r["drug_id"]))
    new_file = not os.path.exists(OUT)
    todo = [(int(r.DRUG_ID), str(r.DRUG_NAME)) for r in resp.sort_values("DRUG_ID").itertuples() if int(r.DRUG_ID) not in done]
    print(f"{len(done)} cached, {len(todo)} to resolve", flush=True)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader(); f.flush()
        for did, name in todo:
            cands = []
            if name in MANUAL: cands.append(MANUAL[name])
            cands.append(name)
            cands += syn.get(did, [])
            cands += [name.replace("-", ""), name.replace("-", " "), name.replace("_", "-"), name.replace("_", " ")]
            seen = set(); cands = [c for c in cands if not (c in seen or seen.add(c))]
            smi, cid, used = None, None, ""
            for c in cands:
                smi, cid = fetch(c)
                time.sleep(0.22)
                if smi:
                    used = c; break
            w.writerow({"drug_id": did, "drug_name": name, "query": used, "cid": cid or "", "smiles": smi or ""})
            f.flush()
            print(did, name, "->", used, cid, (smi or "")[:60], flush=True)
    with open(OUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_ok = sum(1 for r in rows if r["smiles"])
    print(f"resolved {n_ok}/{len(rows)}; missing: {[r['drug_name'] for r in rows if not r['smiles']]}", flush=True)
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
