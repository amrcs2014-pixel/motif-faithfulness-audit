"""Author-curated pharmacophores of the 30 probe drugs (second, literature-based motif ground truth) and the
positive-control biomarker pairs (gene-level ground truth with a known direction).

PHARMACOPHORE: drug name -> (SMARTS of the substructure that carries the primary target interaction, confidence,
one-line rationale). Confidence 'high' = textbook pharmacophore of the chemotype (e.g. the 4-anilinoquinazoline hinge
binder of EGFR/ERBB2 inhibitors); 'medium' = plausible from the class; 'low' = uncertain, used only in the all-drugs variant.

BIOMARKERS: (drug name, gene, feature block, expected direction) where direction = 'sensitive' (feature high /
present -> lower ln IC50) or 'resistant' (feature present -> higher ln IC50). Feature blocks: expr (log TPM),
mut (damaging or hotspot mutation), cnv (log2 relative copy number).
"""

PHARMACOPHORE = {
    "Navitoclax": ("C(=O)NS(=O)(=O)c", "high", "acylsulfonamide of the BH3-mimetic (BCL-2/BCL-XL groove binder, ABT-737 class)"),
    "Nilotinib": ("Nc1nccc(-c2cccnc2)n1", "high", "2-anilino-4-(pyridin-3-yl)pyrimidine hinge binder (imatinib/nilotinib class)"),
    "Bosutinib": ("N#Cc1cnc2ccccc2c1N", "high", "4-anilino-3-quinolinecarbonitrile hinge binder (SRC/ABL)"),
    "Axitinib": ("c1ccc2cnnc2c1", "high", "indazole hinge binder (VEGFR)"),
    "AZD7762": ("NC(=O)Nc1ccsc1C(=O)N", "high", "3-ureido-thiophene-2-carboxamide hinge binder (CHK1)"),
    "JNK Inhibitor VIII": ("N#Cc1c(N)cc(NC=O)nc1", "medium", "2-acylamino-4-amino-3-cyanopyridine core (JNK)"),
    "Wee1 Inhibitor": ("O=c1ccc2cnc(N)nc2n1", "high", "2-anilino-pyrido[2,3-d]pyrimidin-7-one hinge binder (PD0166285)"),
    "Nutlin-3a (-)": ("C1(=NC(c2ccc(Cl)cc2)C(N1)c1ccc(Cl)cc1)", "high", "cis-4,5-bis(4-chlorophenyl)imidazoline filling the MDM2 Leu26/Trp23 pockets"),
    "ZM447439": ("Nc1ncnc2ccccc12", "high", "4-anilinoquinazoline hinge binder (Aurora)"),
    "Selumetinib": ("Brc1ccc(N)c(Cl)c1", "high", "4-bromo-2-chloroaniline of the allosteric MEK pocket (PD0325901 class)"),
    "EHT-1864": ("c1ccc2ncccc2c1", "low", "quinoline; Rac-binding element not established"),
    "Irinotecan": ("CC[C;R](O)C(=O)O[CH2;R]", "high", "camptothecin E-ring alpha-hydroxy lactone (TOP1-DNA covalent complex)"),
    "Rucaparib": ("[#6;R](=O)[NH;R][CH2;R][CH2;R]", "high", "cyclic lactam nicotinamide mimic (PARP1 catalytic site)"),
    "Bortezomib": ("[B](O)O", "high", "boronic acid warhead (proteasome beta-5 threonine)"),
    "Tamoxifen": ("C(c)(c)=C(c)", "high", "triarylethylene scaffold (ER ligand-binding domain)"),
    "EPZ004777": ("Nc1ncnc2nccc12", "high", "4-aminopyrrolo[2,3-d]pyrimidine (adenosine mimic in the DOT1L SAM site)"),
    "Pevonedistat": ("NS(=O)(=O)OC", "high", "sulfamate forming the covalent NEDD8-adduct (NAE)"),
    "Lapatinib": ("Nc1ncnc2ccccc12", "high", "4-anilinoquinazoline hinge binder (EGFR/ERBB2)"),
    "LGK974": ("O=CNc1ccc(cn1)-c1cnccn1", "medium", "2-acylamino-5-(pyrazin-2-yl)pyridine of the PORCN inhibitor class"),
    "PCI-34051": ("C(=O)NO", "high", "hydroxamic acid zinc binder (HDAC8)"),
    "Afuresertib": ("NC(=O)c1sccc1", "medium", "thiophene-2-carboxamide hinge binder (AKT ATP site)"),
    "AGI-5198": ("Cc1nccn1CC(=O)N", "low", "2-methylimidazolyl acetamide; IDH1 R132H allosteric element uncertain"),
    "Cediranib": ("c1ncnc2ccccc12", "high", "4-aryloxyquinazoline hinge binder (VEGFR)"),
    "GDC0810": ("C(c)(c)=C(c)", "high", "triarylethylene scaffold with acrylic acid (SERD)"),
    "I-BRD9": ("CCn1ccc2sccc2c1=O", "low", "N-ethyl thienopyridinone acetyl-lysine mimic (BRD9); uncertain"),
    "NVP-ADW742": ("Nc1ncnc2nccc12", "high", "4-aminopyrrolo[2,3-d]pyrimidine hinge binder (IGF1R)"),
    "WIKI4": ("O=C1[NX3]C(=O)c2cccc3cccc1c23", "high", "naphthalimide nicotinamide mimic (tankyrase)"),
    "BIBR-1532": ("OC(=O)c1ccccc1NC(=O)", "medium", "N-acyl anthranilic acid (telomerase non-competitive site)"),
    "AMG-319": ("c1ncnc2ncnc12", "high", "purine hinge binder (PI3K-delta)"),
    "SGC0946": ("Nc1ncnc2nccc12", "high", "4-aminopyrrolo[2,3-d]pyrimidine (DOT1L SAM site, EPZ004777 analogue)"),
}

BIOMARKERS = [
    ("Lapatinib", "ERBB2", "expr", "sensitive"),
    ("Lapatinib", "ERBB2", "cnv", "sensitive"),
    ("Lapatinib", "EGFR", "expr", "sensitive"),
    ("Nutlin-3a (-)", "TP53", "mut", "resistant"),
    ("Nutlin-3a (-)", "MDM2", "expr", "sensitive"),
    ("Selumetinib", "BRAF", "mut", "sensitive"),
    ("Selumetinib", "KRAS", "mut", "sensitive"),
    ("Tamoxifen", "ESR1", "expr", "sensitive"),
    ("GDC0810", "ESR1", "expr", "sensitive"),
    ("Navitoclax", "BCL2", "expr", "sensitive"),
    ("Afuresertib", "PIK3CA", "mut", "sensitive"),
    ("Wee1 Inhibitor", "TP53", "mut", "sensitive"),
]
