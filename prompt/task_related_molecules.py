TASK_RELATED_MOLECULES = {
    # Boltz binding affinity tasks - placeholder reference molecules from YAML templates
    'boltz/CA2': [],  # L-Tyrosine
    'boltz/CDK2': [],
    'boltz/DHFR': [],
    'boltz/FABP4': [],
    'boltz/JNK1': [],
    'boltz/P38': [],
    'boltz/THROMBIN': [],
    'boltz/TYK2': [],
    # PMO tasks with reference SMILES
    'pmo/Amlodipine_MPO': [
        'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\\C(=O)OC',
    ],
    'pmo/Celecoxib_Rediscovery': [
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',
    ],
    'pmo/Fexofenadine_MPO': [
        'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4',
    ],
    'pmo/Median1': [
        'CC1(C)C2CCC1(C)C(=O)C2',  # camphor
        'CC(C)C1CCC(C)CC1O',  # menthol
    ],
    'pmo/Median2': [
        'O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C',  # tadalafil
        'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C',  # sildenafil
    ],
    'pmo/Osimertinib_MPO': [
        'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34',
    ],
    'pmo/Perindopril_MPO': [
        'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC',
    ],
    'pmo/Ranolazine_MPO': [
        'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2',
    ],
    'pmo/Sitagliptin_MPO': [
        'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F',
    ],
    'pmo/Zaleplon_MPO': [
        'O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1',
    ],
    'pmo/Albuterol_similarity': [
        'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
    ],
    'pmo/Deco_hop': [
        'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C',  # reference pharmacophore
    ],
    'pmo/Mestranol_similarity': [
        'COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1',
    ],
    'pmo/Scaffold_hop': [
        'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C',  # reference pharmacophore
    ],
    'pmo/Thiothixene_Rediscovery': [
        'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1',
    ],
    'pmo/Troglitazone_Rediscovery': [
        'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O',
    ],
    # Tasks without reference SMILES (constraint-based)
    'pmo/DRD2': [],
    'pmo/JNK3': [],
    'pmo/GSK3B': [],
    'pmo/QED': [],
    'pmo/Isomers_c7h8n2o2': [],
    'pmo/Isomers_c9h10n2o2pf2cl': [],
    'pmo/Valsartan_smarts': [],
}
