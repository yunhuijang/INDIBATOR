import numpy as np
import src.evaluate.pmo.auc as auc
from rdkit import Chem
import sys
import os
from rdkit.Chem import RDConfig
from src.evaluate.pmo.auc import top_auc

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def compute_metrics(smiles_score_list:list, freq_log: int):
    
    max_oracle_calls = 1000
    
    finish = len(smiles_score_list) >= max_oracle_calls
    if finish:
        smiles_score_list = smiles_score_list[:max_oracle_calls]
    temp_top100 = smiles_score_list[:100]
    smis = [item[0] for item in temp_top100]
    scores = [item[1] for item in temp_top100]
    n_calls = len(smiles_score_list)
    

    avg_top1 = np.max(scores)
    avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
    avg_top100 = np.mean(scores)
    avg_sa = np.mean([sascorer.calculateScore(Chem.MolFromSmiles(smi)) for smi in smis])

    # Construct mol_buffer
    seen = set()
    mol_buffer = {}
    for i, (smi, score) in enumerate(smiles_score_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:  # valid SMILES
            canonical_smi = Chem.MolToSmiles(mol)  # Canonicalize
            if canonical_smi not in seen:
                # Store as {canonical_smi: [score, generation index]}
                mol_buffer[canonical_smi] = [score, i + 1]
                seen.add(canonical_smi)
            else:
                continue  # duplicated canonical SMILES
        else:
            continue  # invalid SMILES
        
    score = top_auc(mol_buffer, 10, finish, freq_log, max_oracle_calls)
    overall_score = {
        "avg_top1": avg_top1, 
        "avg_top10": avg_top10, 
        "avg_top100": avg_top100, 
        "auc_top1": top_auc(mol_buffer, 1, finish, freq_log, max_oracle_calls),
        "auc_top10": score,
        "auc_top100": top_auc(mol_buffer, 100, finish, freq_log, max_oracle_calls),
        "avg_sa": avg_sa,
        "n_oracle": n_calls,
        "score": score
    }

    return overall_score

def compute_pmo(oracle, smiles:list, freq_log: int):
    """
    Compute the PMO for a given task description.
    """
    # Remove duplicates
    smiles =list(set(smiles))
    
    smiles_score_list = [
        (smi, oracle(smi)) if validate_smiles(smi) else (smi, 0)
        for smi in smiles
    ]
    
    smiles_score_list = sorted(smiles_score_list, key=lambda x: x[1], reverse=True)
    
    result = compute_metrics(
        smiles_score_list,
        freq_log
    )
    return result, smiles_score_list