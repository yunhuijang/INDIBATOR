"""Reviewer agent for quality verification of molecular candidates."""

import logging
from typing import Dict, List, Any
import wandb
import json
import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


from src.evaluate.tools_evaluate import evaluate_pmo, evaluate_lead, evaluate_boltz
from src.agent.state import MoleculeCandidate, ScientistProfile
from src.utils import extract_content

logger = logging.getLogger(__name__)

REVIEWER_PROMPT = """You are the Reviewer Agent responsible for quality verification of molecular candidates.

Your role is to evaluate each candidate molecule for:
1. Chemical validity - Is the SMILES string valid?
2. Drug-likeness - Does it follow Lipinski's rule of five?
3. Synthetic accessibility - Can it be synthesized practically?
4. Predicted activity - How likely is it to achieve the task goal?

Use the compute_molecule_score tool to get objective metrics for each candidate.
After scoring, provide a final ranked list with justifications.

Be thorough and objective in your assessments.
"""


def review_candidates(
    model,
    candidates: List[MoleculeCandidate],
    task_name: str,
    seed_mol_index: int,
    sim_threshold: float,
    freq_log: int,
    scientist_profiles: Dict[str, ScientistProfile],
    run_name: str,
    num_candidates: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    
    """Review and score all candidate molecules.

    Args:
        model: The LLM model to use
        candidates: List of candidate molecules from debate
        task_name: The optimization task name
        seed_mol_index: Seed molecule index for lead optimization
        sim_threshold: Similarity threshold for lead optimization

    Returns:
        overall_score: dict with aggregated metrics
        final_output: List of ALL candidates with scores and qualification flags,
                      ranked with qualified+valid_ds first
    """
    logger.info(f"Reviewing {len(candidates)} candidates...")

    scored_candidates = []
    smiles_list = [c["smiles"] for c in candidates]
    final_output = []
    # Lead optimization tasks
    if 'lead_optimization' in task_name:
        overall_score, result_df = evaluate_lead(smiles_list, task_name, seed_mol_index, sim_threshold)
        result_dict = result_df.set_index('smiles').to_dict(orient='index')

        for candidate in candidates:
            smiles = candidate["smiles"]
            scientist_profile = scientist_profiles.get(candidate["proposer"], {'name': candidate["proposer"], 'publications': [], 'molecules': []})
            molecules = scientist_profile.get('molecules', [])
            molecules_smiles = [m['smiles'] for m in molecules]
            is_novel = smiles not in molecules_smiles
            if smiles in result_dict:
                info = result_dict[smiles]
                candidate["score"] = info.get("ds") or 0
                candidate["score_details"] = {'qed': info.get("qed"), 'sa': info.get("sa"), 'sim': info.get("sim"), 'is_novel': is_novel}
                candidate["is_valid"] = info.get("is_valid", False)
                candidate["is_qualified"] = info.get("is_qualified", False)
                candidate["has_valid_ds"] = info.get("has_valid_ds", False)
                scored_candidates.append(candidate)
        # Sort: qualified+valid_ds first (by score desc), then others (by score desc)
        qualified_valid = [c for c in scored_candidates if c.get("is_qualified") and c.get("has_valid_ds")]
        others = [c for c in scored_candidates if not (c.get("is_qualified") and c.get("has_valid_ds"))]

        qualified_valid.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
        others.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)

        all_sorted = qualified_valid + others
        
        for i, c in enumerate(all_sorted):
            final_output.append({
                "rank": i + 1,
                "smiles": c["smiles"],
                "score": c.get("score", 0) or 0,
                "proposer": c["proposer"],
                "debate_round": c["round"],
                "debate_votes": c.get("avg_vote", 0),
                "num_votes": len(c.get("votes", {})),
                "score_details": c.get("score_details", {}),
                "is_valid": c.get("is_valid", False),
                "is_qualified": c.get("is_qualified", False),
                "has_valid_ds": c.get("has_valid_ds", False),
            })

        top_score = final_output[0]['score'] if final_output else 0
        logger.info(f"Review complete. Top score: {top_score}")
    elif 'boltz' in task_name:
        # Boltz binding affinity tasks
        smiles_list = [c['smiles'] for c in candidates]
        wandb.log({"smiles_list": smiles_list})
        folder_name =f"output/boltz/{task_name.split('/')[-1]}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
        json.dump(smiles_list, open(f"{folder_name}/smiles_list_{run_name}.json", "w"))
        
        if len(candidates) >= num_candidates:
            overall_score, result_df = evaluate_boltz(candidates, task_name, run_name)
            result_dict = result_df.set_index('smiles').to_dict(orient='index')

            for candidate in candidates:
                smiles = candidate["smiles"]
                scientist_profile = scientist_profiles.get(candidate["proposer"], {'name': candidate["proposer"], 'publications': [], 'molecules': []})
                molecules = scientist_profile.get('molecules', [])
                molecules_smiles = [m['smiles'] for m in molecules]
                is_novel = smiles not in molecules_smiles

                if smiles in result_dict:
                    info = result_dict[smiles]
                    # Score is negated affinity (higher = better binding)
                    candidate["score"] = info.get("score", 0)
                    candidate["score_details"] = {
                        'affinity_pred_value': info.get("affinity_pred_value", 0),
                        'affinity_probability_binary': info.get("affinity_probability_binary", 0),
                        'is_novel': is_novel
                    }
                    candidate["is_valid"] = info.get("is_valid", False)
                    scored_candidates.append(candidate)

            # Sort by score descending (higher score = better binding)
            scored_candidates.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
            all_sorted = scored_candidates

            for i, c in enumerate(all_sorted):
                final_output.append({
                    "rank": i + 1,
                    "smiles": c["smiles"],
                    "score": c.get("score", 0) or 0,
                    "proposer": c["proposer"],
                    "debate_round": c["round"],
                    "debate_votes": c.get("avg_vote", 0),
                    "num_votes": len(c.get("votes", {})),
                    "score_details": c.get("score_details", {}),
                    "is_valid": c.get("is_valid", False),
                })

            top_score = final_output[0]['score'] if final_output else 0
            logger.info(f"Review complete. Top boltz score: {top_score}")
        else:
            # Not enough candidates yet for boltz evaluation
            all_sorted = candidates
            overall_score = {"score": 0}
            for i, c in enumerate(candidates):
                final_output.append({
                    "rank": i + 1,
                    "smiles": c["smiles"],
                    "score": c.get("score", 0) or 0,
                    "proposer": c["proposer"],
                    "debate_round": c["round"],
                    "debate_votes": c.get("avg_vote", 0),
                    "num_votes": len(c.get("votes", {})),
                    "score_details": c.get("score_details", {}),
                    "is_valid": c.get("is_valid", False),
                })
            logger.info(f"Boltz: only {len(candidates)} candidates, need {num_candidates} for evaluation")
    elif 'pmo' in task_name:
        # PMO tasks
        task_name_clean = task_name.split('/')[-1]
        overall_score, result_df = evaluate_pmo(smiles_list, task_name_clean, freq_log)
        result_dict = result_df.set_index('smiles').to_dict(orient='index')
        for candidate in candidates:
            smiles = candidate["smiles"]
            scientist_profile = scientist_profiles.get(candidate["proposer"], {'name': candidate["proposer"], 'publications': [], 'molecules': []})
            molecules = scientist_profile.get('molecules', [])
            molecules_smiles = [m['smiles'] for m in molecules]
            is_novel = smiles not in molecules_smiles
            
            if smiles in molecules_smiles:
                candidate["is_selected"] = True
            else:
                candidate["is_selected"] = False
            if smiles in result_dict:
                info = result_dict[smiles]
                candidate["score"] = info.get("score")
                candidate["score_details"] = {
                    "is_novel": is_novel,
                }
                scored_candidates.append(candidate)
        all_sorted = scored_candidates
        for i, c in enumerate(all_sorted):
            final_output.append({
                "rank": i + 1,
                "smiles": c["smiles"],
                "score": c.get("score", 0) or 0,
                "proposer": c["proposer"],
                "debate_round": c["round"],
                "debate_votes": c.get("avg_vote", 0),
                "num_votes": len(c.get("votes", {})),
                "score_details": c.get("score_details", {}),
            })
        
        logger.info(f"Review complete. TOP 10 AUC score: {overall_score['score']:.4f}")
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    overall_score['avg_novel'] = sum([c.get("score_details", {}).get("is_novel", False) for c in all_sorted]) / len(all_sorted)
    return overall_score, final_output
