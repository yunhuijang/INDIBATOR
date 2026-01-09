from langchain_core.tools import tool
import wandb
import re
from tdc import Oracle

from src.evaluate.lead.utils_lead import compute_lead
from src.evaluate.pmo.utils_pmo import compute_pmo



def evaluate_lead(smiles: list, task_name: str, seed_idx: int, sim_threshold: float):
    """
    Evaluate the lead optimization for a given task following GenMol methodology.

    Args:
        smiles: List of SMILES strings to evaluate
        task_name: Target protein name (parp1, fa7, 5ht1b, braf, jak2)
        seed_idx: Seed molecule index (0, 1, or 2)
        sim_threshold: Minimum Tanimoto similarity to seed (0.4 or 0.6)

    Returns:
        Evaluation results dict with best docking score among qualifying molecules
    """
    if task_name not in ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']:
        raise ValueError(f'Wrong protein name: {task_name}. You can only choose from parp1, fa7, 5ht1b, braf, jak2.')

    overall_score, result_df = compute_lead(task_name, smiles, seed_idx, sim_threshold)

    return overall_score, result_df


def evaluate_pmo(smiles: list, task_name:str=None):
    """
    Evaluate the PMO for a given task description.
    """
    task_list = ['Amlodipine_MPO', 'Celecoxib_Rediscovery', 'DRD2', 'Fexofenadine_MPO',
                 'JNK3', 'GSK3B', 'Median 1', 'Median 2', 'Osimertinib_MPO', 'Perindopril_MPO',
                 'Ranolazine_MPO', 'Sitagliptin_MPO', 'Zaleplon_MPO']
    
    if task_name not in task_list:
        raise ValueError('Wrong task name: {task_name}. You can only choose from {task_list}.')
    
    oracle = Oracle(name=task_name)
    result = compute_pmo(oracle, smiles)
    
    wandb.log