import numpy as np
import src.evaluate.pmo.auc as auc
from rdkit import Chem

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def compute_metrics(smiles_score_list:list, iteration:int, max_iterations:int):
    finish = iteration + 1 >= max_iterations
    auc_top10_all, _ = auc.compute_topk_auc(
        smiles_score_list,
        top_k=10,
        max_oracle_calls=1000,
        freq_log=1,
        buffer_max_idx=iteration + 1,
        finish=finish,
    )
    auc_top1_all, _ = auc.compute_topk_auc(
        smiles_score_list,
        top_k=1,
        max_oracle_calls=1000,
        freq_log=1,
        buffer_max_idx=iteration + 1,
        finish=finish,
    )

    sorted_all = sorted(smiles_score_list, key=lambda x: x[1], reverse=True)
    top_10_all = sum(score for _, score in sorted_all[:10]) / 10

    return {
        "top_10_avg_score_all": top_10_all,
        "auc_top1_all": auc_top1_all,
        "auc_top10_all": auc_top10_all,
    }

def compute_pmo(oracle, smiles:list, iteration:int, max_iterations:int):
    """
    Compute the PMO for a given task description.
    """
    
    smiles_score_list = [
        (smi, oracle(smi)) if validate_smiles(smi) else (smi, 0)
        for smi in smiles
    ]
    
    result = compute_metrics(
        smiles_score_list, iteration, max_iterations
    )
    return result