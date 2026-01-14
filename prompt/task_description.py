
lead_optimization_prompt = """
Your task is to design a SMILES string for a molecule that optimizes binding affinity to {protein_name}.
You must start from the provided seed molecule, modify it to improve predicted binding affinity (docking score), while strictly satisfying all drug-likeness and similarity constraints.

# Seed Molecule: {seed_mol}

# Conditions:
- Optimize binding affinity (docking score) to {protein_name} protein.
- Maintain structural similarity to the chosen seed molecule (Tanimoto similarity >= {sim_threshold}).
- Design drug-like molecules with QED (Quantitative Estimation of Drug-likeness) >= 0.6.
- Ensure synthetic accessibility with SA score <= 4.

# IMPORTANT CONSTRAINTS:
- The generated molecule MUST be structurally similar to the seed molecule, drug-like, and synthetically accessible. Keep the conditions above in mind when generating the molecule. This is very important.
- YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO THE SEED.
- Avoid repeating molecules you already generated.

# Suggested modification strategies (optional guidance)
- Consider conservative bioisosteric replacements, small substituent additions/deletions, or ring modifications that preserve scaffold similarity but may improve docking score.
- Avoid adding large, complex, or polycyclic groups that harm SA score.
- Check that logP, molecular weight, and polar surface area remain in drug-like ranges.
"""

TASK_DESCRIPTION = {
    # Lead optimization task description
    'lead_optimization/parp1': lead_optimization_prompt,
    'lead_optimization/fa7': lead_optimization_prompt,
    'lead_optimization/5ht1b': lead_optimization_prompt,
    'lead_optimization/braf': lead_optimization_prompt,
    'lead_optimization/jak2': lead_optimization_prompt,
    'pmo/Amlodipine_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions: 
    
    # Conditions:
    - Achieve high structural similarity to amlodipine (SMILES: 'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC').
    - Preferably maintain around 3 rings in the molecular structure to preserve desired complexity.

    # IMPORTANT CONSTRAINTS:  
    YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO AMLODIPINE: 'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC'.
    """,
    # PMO task description
    'pmo/Celecoxib_Rediscovery': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition: 

    # Conditions:
    Design a drug-like molecule structurally similar to celecoxib (SMILES: CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F). 
    Preserve the core scaffold and important pharmacophores. Celecoxib contains: \n{celecoxib_functional_group}.

    # IMPORTANT CONSTRAINTS:  
    YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO CELECOXIB: 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'.
    """,
    
    'pmo/DRD2':"""
    Your task is to design a SMILES string for a molecule that satisfies the following condition: 
    
    # Conditions:
    Maximize the probability of binding to the DRD2 receptor (Dopamine Receptor D2).

    # IMPORTANT CONSTRAINTS:
    - Design drug-like molecules.
    - Maximize the DRD2 binding score as high as possible.
    - Avoid generating identical structures to provided examples.
    - Avoid repeating molecules you already generated.
    """,
    
    'pmo/Fexofenadine_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:
    # Conditions:
    - Achieve high structural similarity to fexofenadine (SMILES: CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4).
    - Target a Topological Polar Surface Area (TPSA) around 90.
    - Aim for moderate lipophilicity with a LogP value close to 4.

    # IMPORTANT CONSTRAINTS:  
    YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO FEXOFENADINE.
    """,
    
    'pmo/JNK3': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule with high predicted JNK3 (c-Jun N-terminal kinase 3) inhibitory activity.
    - Consider structural features known to enhance kinase inhibition.

    # IMPORTANT CONSTRAINTS:
    - Design drug-like molecules with favorable ADMET properties.
    - Maximize the JNK3 inhibitory activity score as high as possible.
    - Avoid generating identical structures to provided examples.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/GSK3B': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule with high predicted GSK3B (Glycogen Synthase Kinase 3 Beta) inhibitory activity.
    - Consider structural features known to enhance kinase binding affinity.

    # IMPORTANT CONSTRAINTS:
    - Design drug-like molecules with favorable ADMET properties.
    - Maximize the GSK3B binding/inhibitory activity score as high as possible.
    - Avoid generating identical structures to provided examples.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Median1': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Design a drug-like molecule that is simultaneously similar to both camphor (SMILES: 'CC1(C)C2CCC1(C)C(=O)C2') and menthol (SMILES: 'CC(C)C1CCC(C)CC1O').
    - Structural similarity is measured based on ECFP4 fingerprints.
    - The goal is to maximize the median similarity to both reference molecules.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO CAMPHOR: 'CC1(C)C2CCC1(C)C(=O)C2'.
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO MENTHOL: 'CC(C)C1CCC(C)CC1O'.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Median2': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Design a drug-like molecule that is simultaneously similar to both tadalafil (SMILES: 'O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C') and sildenafil (SMILES: 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C').
    - Structural similarity is measured based on ECFP4 fingerprints.
    - The goal is to maximize the median similarity to both reference molecules.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TADALAFIL: 'O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C'.
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO SILDENAFIL: 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C'.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Osimertinib_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Achieve high structural similarity to osimertinib (SMILES: 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34').
    - Target a Topological Polar Surface Area (TPSA) around 100.
    - Aim for low lipophilicity with a LogP value close to 1.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO OSIMERTINIB: 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Perindopril_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Achieve high structural similarity to perindopril (SMILES: 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC').
    - Preferably maintain around 2 aromatic rings in the molecular structure.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO PERINDOPRIL: 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC'.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Ranolazine_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Achieve high structural similarity to ranolazine (SMILES: 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2').
    - Target a Topological Polar Surface Area (TPSA) around 95.
    - Aim for lipophilicity with a LogP value close to 7.
    - Preferably include around 1 fluorine atom in the structure.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO RANOLAZINE: 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Sitagliptin_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Achieve high structural similarity to sitagliptin (SMILES: 'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F').
    - The molecule must have the exact molecular formula: C16H15F6N5O.
    - Maintain similar LogP and TPSA values to sitagliptin.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO SITAGLIPTIN: 'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F'.
    - The molecular formula MUST be exactly C16H15F6N5O.
    - Avoid repeating molecules you already generated.
    """,
    'pmo/Zaleplon_MPO': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Achieve high structural similarity to zaleplon (SMILES: 'O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1').
    - The molecule must have the exact molecular formula: C19H17N3O2.
    - Structural similarity is measured based on atom-pair fingerprints.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ZALEPLON: 'O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1'.
    - The molecular formula MUST be exactly C19H17N3O2.
    - Avoid repeating molecules you already generated.
    """,
}