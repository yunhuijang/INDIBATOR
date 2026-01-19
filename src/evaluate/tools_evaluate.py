from langchain_core.tools import tool
import wandb
import re
from tdc import Oracle
import pandas as pd
import json

from src.evaluate.lead.utils_lead import compute_lead, compute_lead_mood
from src.evaluate.pmo.utils_pmo import compute_pmo
from src.evaluate.boltz.utils_boltz import compute_boltz



def evaluate_lead(smiles: list, task_name: str, seed_idx: int, sim_threshold: float):
    """
    Evaluate the lead optimization for a given task following GenMol methodology.

    Args:
        smiles: List of SMILES strings to evaluate
        task_name: Target protein name (parp1, fa7, 5ht1b, braf, jak2, sars_cov_2)
        seed_idx: Seed molecule index (0, 1, or 2)
        sim_threshold: Minimum Tanimoto similarity to seed (0.4 or 0.6)

    Returns:
        Evaluation results dict with best docking score among qualifying molecules
    """
    task_name_clean = task_name.split('/')[-1]
    if task_name_clean not in ['parp1', 'fa7', '5ht1b', 'braf', 'jak2', 'sars_cov_2']:
        raise ValueError(f'Wrong protein name: {task_name}. You can only choose from parp1, fa7, 5ht1b, braf, jak2, sars_cov_2.')

    
    if task_name.startswith('lead_optimization_mood'):
        overall_score, result_df = compute_lead_mood(task_name_clean, smiles)
    else:
        overall_score, result_df = compute_lead(task_name_clean, smiles, seed_idx, sim_threshold)

    return overall_score, result_df


def evaluate_pmo(smiles: list, task_name:str, freq_log: int):
    """
    Evaluate the PMO for a given task description.
    """
    # TDC oracle names are lowercase
    task_list = ['amlodipine_mpo', 'celecoxib_rediscovery', 'drd2', 'fexofenadine_mpo',
                 'jnk3', 'gsk3b', 'median1', 'median2', 'osimertinib_mpo', 'perindopril_mpo',
                 'ranolazine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo',
                 # new tasks
                 'albuterol_similarity', 'valsartan_smarts', 'thiothixene_rediscovery', 
                 'troglitazone_rediscovery', 'qed', 'mestranol_similarity', 'scaffold_hop', 
                 'deco_hop', 'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl']

    # Convert to lowercase for case-insensitive matching
    task_name_lower = task_name.lower() if task_name else None

    if task_name_lower not in task_list:
        raise ValueError(f'Wrong task name: {task_name}. You can only choose from {task_list}.')

    oracle = Oracle(name=task_name_lower)
    result, smiles_score_list = compute_pmo(oracle, smiles, freq_log)

    result_df = pd.DataFrame(smiles_score_list, columns=['smiles', 'score'])
    return result, result_df


def evaluate_boltz(candidates: list, task_name: str, run_name: str):
    """
    Evaluate molecules using Boltz binding affinity prediction.

    Args:
        smiles: List of SMILES strings to evaluate
        task_name: Task name in format 'boltz/PROTEIN' (e.g., 'boltz/CA2')

    Returns:
        overall_score: dict with aggregated metrics
        result_df: DataFrame with per-molecule results
    """
    protein = task_name.split('/')[-1].upper()
    valid_proteins = ['CA2', 'CDK2', 'DHFR', 'FABP4', 'JNK1', 'P38', 'THROMBIN', 'TYK2']

    if protein not in valid_proteins:
        raise ValueError(f'Invalid protein: {protein}. Must be one of {valid_proteins}.')

    smiles_list = [c['smiles'] for c in candidates]
    wandb.log({"smiles_list": smiles_list})
    json.dump(smiles_list, open(f"output/boltz/{protein}/smiles_list_{run_name}.json", "w"))

    overall_score, result_df = compute_boltz(protein, candidates, run_name)
    return overall_score, result_df