"""Reviewer agent for quality verification of molecular candidates."""

import logging
from typing import Dict, List, Any
import wandb

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from src.evaluate.tools_evaluate import evaluate_pmo, evaluate_lead
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


# def create_reviewer_agent(model):
#     """Create the reviewer agent for quality verification.

#     Args:
#         model: The LLM model to use

#     Returns:
#         Compiled ReAct agent for reviewing molecules
#     """
#     agent = create_agent(
#         model=model,
#         tools=[evaluate_lead, evaluate_pmo],
#         system_prompt=REVIEWER_PROMPT,
#     )

#     return agent


def review_candidates(
    model,
    candidates: List[MoleculeCandidate],
    task_name: str,
    seed_mol_index: int,
    sim_threshold: float,
    freq_log: int,
    scientist_profiles: List[ScientistProfile]
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
    task_name_clean = task_name.split('/')[-1]
    final_output = []
    # Lead optimization tasks
    if task_name.startswith('lead_optimization'):
        overall_score, result_df = evaluate_lead(smiles_list, task_name_clean, seed_mol_index, sim_threshold)
        result_dict = result_df.set_index('smiles').to_dict(orient='index')

        for candidate in candidates:
            smiles = candidate["smiles"]
            scientist_profile = scientist_profiles.get([candidate["proposer"]], {'name': candidate["proposer"], 'publications': [], 'molecules': []})
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
    else:
        # PMO tasks
        overall_score, result_df = evaluate_pmo(smiles_list, task_name_clean, freq_log)
        result_dict = result_df.set_index('smiles').to_dict(orient='index')
        for candidate in candidates:
            smiles = candidate["smiles"]
            scientist_profile = scientist_profiles.get([candidate["proposer"]], {'name': candidate["proposer"], 'publications': [], 'molecules': []})
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
    overall_score['avg_novel'] = sum([c.get("score_details", {}).get("is_novel", False) for c in all_sorted]) / len(all_sorted)
    return overall_score, final_output
    # Prepare final output with rank and qualification flags



# def run_reviewer_agent(
#     model,
#     candidates: List[MoleculeCandidate],
#     task_description: str
# ) -> List[Dict[str, Any]]:
#     """Run the reviewer agent for comprehensive evaluation.

#     This version uses the LLM agent to provide qualitative analysis
#     in addition to the quantitative scores.

#     Args:
#         model: The LLM model to use
#         candidates: List of candidate molecules
#         task_description: The optimization task

#     Returns:
#         List of scored and analyzed molecules
#     """
#     # First, get quantitative scores
#     scored_candidates = review_candidates(model, candidates, task_description)

#     if not scored_candidates:
#         return []

#     # Optionally, get qualitative analysis from the agent
#     agent = create_reviewer_agent(model)

#     # Format candidates for agent
#     candidates_text = "\n".join([
#         f"{c['rank']}. {c['smiles']} (score: {c['score']:.3f}, debate votes: {c['debate_votes']:.2f})"
#         for c in scored_candidates[:10]
#     ])

#     prompt = f"""Task: {task_description}

# Here are the top candidate molecules from the debate, with their scores:

# {candidates_text}

# Please provide:
# 1. A brief analysis of why the top candidates scored well
# 2. Any concerns about the top candidates
# 3. Your recommendation for the best molecule to pursue

# Use the compute_molecule_score tool if you need to verify any scores.
# """

#     try:
#         result = agent.invoke({"messages": [HumanMessage(content=prompt)]})

#         # Extract analysis from result
#         analysis = extract_content(result)

#         # Add analysis to output
#         if analysis:
#             for c in scored_candidates:
#                 c["reviewer_analysis"] = analysis

#     except Exception as e:
#         logger.warning(f"Agent analysis failed: {e}")

#     return scored_candidates



