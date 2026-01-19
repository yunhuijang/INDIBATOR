
lead_optimization_prompt = """
Your task is to design a SMILES string for a molecule that optimizes binding affinity to {protein_name}.
You must start from the provided seed molecule, modify it to improve predicted binding affinity (docking score), while strictly satisfying all drug-likeness and similarity constraints.

# Seed Molecule: {seed_mol}

# Hard Constraints (MUST satisfy ALL):
1. Tanimoto Similarity >= {sim_threshold}
2. QED (drug-likeness) >= 0.6
3. SA Score (synthetic accessibility) <= 4
4. NOT identical to seed molecule

# Metric Definitions:

## Tanimoto Similarity (target: >= {sim_threshold})
Measures structural similarity between your molecule and the seed using Morgan fingerprints.
- Score range: 0 (completely different) to 1 (identical)
- Calculated as: (shared features) / (total unique features in both molecules)
- To maintain high similarity: preserve the core scaffold and make only small modifications

## QED - Quantitative Estimate of Drug-likeness (target: >= 0.6)
A composite score combining 8 drug-like properties, ranging 0-1 (higher = more drug-like).
Based on: molecular weight, logP, H-bond donors/acceptors, polar surface area, rotatable bonds, aromatic rings, and structural alerts.
To achieve QED >= 0.6:
- Molecular Weight: 200-500 Da
- LogP: 0-5
- H-bond donors: <= 5
- H-bond acceptors: <= 10
- Rotatable bonds: <= 10
- Aromatic rings: 1-4

## SA Score - Synthetic Accessibility (target: <= 4)
Estimates how easy a molecule is to synthesize, ranging 1 (easy) to 10 (very difficult).
Calculated from fragment contributions (based on 1M PubChem molecules) plus complexity penalties for unusual features.
To achieve SA <= 4:
- Use common, commercially available building blocks
- Avoid: large rings (>8 atoms), bridgehead/spiro atoms, multiple stereocenters
- Prefer simple ring systems (benzene, pyridine, piperidine)

# Recommended Modifications (preserve similarity, maintain drug-likeness):
- Small substituent changes: -H → -F, -CH3 → -CF3, -OH → -OCH3
- Bioisosteric replacements: benzene ↔ pyridine, -COOH ↔ -CONH2
- Methylation of amines: -NH2 → -NHCH3
- Small ring modifications: 6-ring → 5-ring

# Modifications to AVOID:
- Adding large polycyclic systems (hurts SA)
- Adding > 2 new rings (may hurt similarity)
- Adding unusual functional groups (hurts SA)
- Removing core scaffold elements (hurts similarity)

Generate ONLY the SMILES string with no explanation.
"""


lead_optimization_mood_prompt = """
Your task is to design a SMILES string for a molecule that optimizes binding affinity to {protein_name}.
You must discover novel molecules that improves the hit ratio, where hit ratio is the number of molecules that satisfy the following constraints divided by the total number of molecules.

# Seed Molecules: {seed_molecules}

# Hard Constraints (MUST satisfy ALL):
1. max(Tanimoto Similarity) <= 0.4
2. QED (drug-likeness) >= 0.5
3. SA Score (synthetic accessibility) <= 5
4. Docking score > {docking_score_threshold}
4. NOT identical to seed molecule

# Metric Definitions:

## Tanimoto Similarity (target: <= 0.4)
Measures structural similarity between your molecule and the seed molecules using Morgan fingerprints.
- Score range: 0 (completely different) to 1 (identical)
- Calculated as: (shared features) / (total unique features in both molecules)
- The maximum Tanimoto Similarity score should be as low as possible.

## QED - Quantitative Estimate of Drug-likeness (target: >= 0.5)
A composite score combining 8 drug-like properties, ranging 0-1 (higher = more drug-like).
Based on: molecular weight, logP, H-bond donors/acceptors, polar surface area, rotatable bonds, aromatic rings, and structural alerts.
To achieve QED >= 0.5:
- Molecular Weight: 200-500 Da
- LogP: 0-5
- H-bond donors: <= 5
- H-bond acceptors: <= 10
- Rotatable bonds: <= 10
- Aromatic rings: 1-4

## SA Score - Synthetic Accessibility (target: <= 5)
Estimates how easy a molecule is to synthesize, ranging 1 (easy) to 10 (very difficult).
Calculated from fragment contributions (based on 1M PubChem molecules) plus complexity penalties for unusual features.
To achieve SA <= 5:
- Use common, commercially available building blocks
- Avoid: large rings (>8 atoms), bridgehead/spiro atoms, multiple stereocenters
- Prefer simple ring systems (benzene, pyridine, piperidine)

# Modifications to AVOID:
- Adding large polycyclic systems (hurts SA)
- Adding unusual functional groups (hurts SA)

Generate ONLY the SMILES string with no explanation.
"""

boltz_prompt = """
Your task is to design a SMILES string for a molecule that maximizes binding affinity to {protein_name}.

# Objective:
Design molecules with high predicted binding affinity (low IC50) to {protein_name}.

# Evaluation Metrics:
- affinity_pred_value: log10(IC50) in μM - LOWER values indicate STRONGER binding

# Protein sequence:
{protein_sequence}

# Molecular Constraints:
- Must be a valid SMILES string

# Guidelines for High Binding Affinity:
- Include appropriate functional groups for hydrogen bonding (amines, hydroxyls, carbonyls)
- Consider hydrophobic contacts with protein binding pocket (aromatic rings, alkyl chains)
- Maintain reasonable molecular weight (300-600 Da)
- Include aromatic rings for pi-stacking interactions
- Consider salt bridges with charged residues (carboxylic acids, amines)

Generate ONLY the SMILES string with no explanation.
"""

TASK_DESCRIPTION = {
    # Lead optimization task description
    'lead_optimization/parp1': lead_optimization_prompt,
    'lead_optimization/fa7': lead_optimization_prompt,
    'lead_optimization/5ht1b': lead_optimization_prompt,
    'lead_optimization/braf': lead_optimization_prompt,
    'lead_optimization/jak2': lead_optimization_prompt,
    'lead_optimization/sars_cov_2': lead_optimization_prompt,
    # MOOD lead optimization task description
    'lead_optimization_mood/parp1': lead_optimization_mood_prompt,
    'lead_optimization_mood/fa7': lead_optimization_mood_prompt,
    'lead_optimization_mood/5ht1b': lead_optimization_mood_prompt,
    'lead_optimization_mood/braf': lead_optimization_mood_prompt,
    'lead_optimization_mood/jak2': lead_optimization_mood_prompt,
    'lead_optimization_mood/sars_cov_2': lead_optimization_mood_prompt,
    # Boltz binding affinity prediction task
    'boltz/CA2': boltz_prompt,
    'boltz/CDK2': boltz_prompt,
    'boltz/DHFR': boltz_prompt,
    'boltz/FABP4': boltz_prompt,
    'boltz/JNK1': boltz_prompt,
    'boltz/P38': boltz_prompt,
    'boltz/THROMBIN': boltz_prompt,
    'boltz/TYK2': boltz_prompt,
    # PMO
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

    # New PMO tasks
    'pmo/Albuterol_similarity': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule structurally similar to albuterol (SMILES: 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1').
    - Preserve the core scaffold and key functional groups.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL: 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1'.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Deco_hop': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Design a drug-like molecule by preserving the fixed core scaffold while modifying peripheral decorations.
    - The molecule MUST contain this scaffold pattern: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
    - Avoid these forbidden motifs: CS([#6])(=O)=O and [#7]-c1ccc2ncsc2c1
    - Maintain 0.85 similarity to reference pharmacophore: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C

    # IMPORTANT CONSTRAINTS:
    - Preserve the required scaffold while exploring decoration diversity.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Isomers_c7h8n2o2': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Create a valid chemical structure that is an isomer of the molecular formula C7H8N2O2.
    - The molecule MUST have EXACTLY: 7 Carbon (C), 8 Hydrogen (H), 2 Nitrogen (N), 2 Oxygen (O) atoms.

    # IMPORTANT CONSTRAINTS:
    - The molecular formula MUST be exactly C7H8N2O2. No missing or extra atoms are allowed.
    - Design scientifically plausible chemical structures.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Isomers_c9h10n2o2pf2cl': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Create a valid chemical structure that is an isomer of the molecular formula C9H10N2O2PF2Cl.
    - The molecule MUST have EXACTLY: 9 Carbon (C), 10 Hydrogen (H), 2 Nitrogen (N), 2 Oxygen (O), 1 Phosphorus (P), 2 Fluorine (F), 1 Chlorine (Cl) atoms.

    # IMPORTANT CONSTRAINTS:
    - The molecular formula MUST be exactly C9H10N2O2PF2Cl. No missing or extra atoms are allowed.
    - Design scientifically plausible chemical structures.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Mestranol_similarity': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule structurally similar to mestranol (SMILES: 'COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1').
    - Preserve the core steroid scaffold, ethinyl group, and methoxy substituent.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO MESTRANOL: 'COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1'.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/QED': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule that maximizes the QED (Quantitative Estimation of Drug-likeness) score.
    - QED scores range from 0 to 1, with higher values indicating better drug-likeness.
    - Consider molecular properties like molecular weight, LogP, number of H-bond donors/acceptors, rotatable bonds, and aromatic rings.

    # IMPORTANT CONSTRAINTS:
    - Aim for QED scores approaching 1.0.
    - Ensure synthetic feasibility and chemical realism.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Scaffold_hop': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Design a drug-like molecule by removing the original scaffold while preserving critical decorations.
    - Remove this scaffold: [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12
    - Preserve this decoration pattern: [#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1
    - Reference pharmacophore: CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C

    # IMPORTANT CONSTRAINTS:
    - Creatively modify the core structure while maintaining pharmacophore similarity.
    - Maintain drug-like properties throughout the scaffold substitution.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Thiothixene_Rediscovery': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule structurally similar to thiothixene (SMILES: 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1').
    - Preserve the thioxanthene core structure and essential pharmacophoric elements.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO THIOTHIXENE: 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1'.
    - Maintain drug-like properties.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Troglitazone_Rediscovery': """
    Your task is to design a SMILES string for a molecule that satisfies the following condition:

    # Conditions:
    - Design a drug-like molecule structurally similar to troglitazone (SMILES: 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O').
    - Preserve the thiazolidinedione ring, chroman moiety, and phenolic hydroxyl group.

    # IMPORTANT CONSTRAINTS:
    - YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO TROGLITAZONE: 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O'.
    - Maintain drug-like properties.
    - Avoid repeating molecules you already generated.
    """,

    'pmo/Valsartan_smarts': """
    Your task is to design a SMILES string for a molecule that satisfies the following conditions:

    # Conditions:
    - Design a drug-like molecule that contains the following SMARTS pattern: CN(C=O)Cc1ccc(c2ccccc2)cc1
    - Target approximate molecular properties:
      - LogP around 2.0
      - TPSA around 95
      - Bertz complexity around 800

    # IMPORTANT CONSTRAINTS:
    - The molecule MUST contain the required structural motif: CN(C=O)Cc1ccc(c2ccccc2)cc1
    - Maintain drug-like properties.
    - Avoid repeating molecules you already generated.
    """,
}