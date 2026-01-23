"""Baseline molecular optimization - single LLM call generation without debate.

This module provides a baseline approach for comparison with the multi-agent debate system.
It generates molecules in batches using a single LLM call without scientist selection or debate.
"""

import argparse
import logging
import sys
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import wandb
import config  # Load .env file before initializing clients

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.callbacks import get_openai_callback

from src.utils import get_task_description, safe_parse_json_list, canonicalize_smiles, extract_smiles
from src.agent.reviewer import review_candidates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for baseline molecule generation."""
    task_name: str
    num_molecules: int = 1000
    batch_size: int = 20
    temperature: float = 0.7
    seed_mol_index: int = 1
    sim_threshold: float = 0.4
    freq_log: int = 100
    top_k: int = 10
    rounds: int = 20


def build_batch_prompt(task_description: str, batch_size: int) -> str:
    """Wrap task description with batch generation instructions.

    Args:
        task_description: The molecular optimization task description
        batch_size: Number of molecules to generate per batch

    Returns:
        Prompt string with batch generation instructions
    """
    return f"""You are a researcher specializing in molecular design and drug discovery.

Task: {task_description}

Generate {batch_size} unique molecules. 

Output format:
[
    {{"SMILES": "SMILES string"}},
    {{"SMILES": "SMILES string"}},
    ...
]

Important:
- Output ONLY the JSON array, no other text or markdown
- Ensure all brackets and quotes are properly closed
- Do NOT wrap in code blocks
- Ensure the SMILES strings are valid and have not proposed in previous proposals
- Do NOT propose the duplicated molecules. Each proposal should be unique.
"""


def parse_smiles_from_response(response: str) -> List[str]:
    """Parse and validate SMILES from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        List of valid, canonicalized SMILES strings
    """
    # Use existing parser
    parsed = safe_parse_json_list(response)

    valid_smiles = []
    for item in parsed:
        smiles = None
        if isinstance(item, dict):
            # Try various key names
            smiles = item.get("SMILES") or item.get("smiles") or item.get("Smiles")
        elif isinstance(item, str):
            smiles = item

        if smiles:
            canonical = canonicalize_smiles(smiles)
            if canonical:
                valid_smiles.append(canonical)

    return valid_smiles


def run_baseline(
    task_description: str,
    model_name: str,
    config: BaselineConfig,
) -> Dict[str, Any]:
    """Run baseline molecule generation.

    Args:
        task_description: Natural language task description
        model_name: LLM model to use
        config: Baseline configuration

    Returns:
        Dict with candidates, messages, cost, and token info
    """
    logger.info("=" * 60)
    logger.info("BASELINE MOLECULAR GENERATION")
    logger.info("=" * 60)
    logger.info(f"Task: {config.task_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Target molecules: {config.num_molecules}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("=" * 60)

    # Initialize the LLM
    if 'gpt' in model_name:
        model = ChatOpenAI(model=model_name, temperature=config.temperature)
    elif 'gemini' in model_name:
        model = ChatGoogleGenerativeAI(model=model_name, temperature=config.temperature)
    elif 'claude' in model_name:
        model = ChatAnthropic(model=model_name, temperature=config.temperature)
    elif 'deepseek' in model_name:
        model = ChatDeepSeek(model=model_name, temperature=config.temperature)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    collected_smiles = set()
    candidates = []
    messages = []
    batch_idx = 0

    with get_openai_callback() as cb:
        while len(collected_smiles) < config.num_molecules and batch_idx < config.rounds:
            batch_idx += 1

            # Build prompt for this batch
            prompt = build_batch_prompt(task_description, config.batch_size)

            # Call LLM
            logger.info(f"Batch {batch_idx}: Requesting {config.batch_size} molecules...")
            response = model.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            smiles_list = extract_smiles(response_text)
            unique_smiles_list = list(set(smiles_list))

            # Deduplicate and collect
            new_unique = 0
            for smiles in unique_smiles_list:
                if smiles not in collected_smiles:
                    collected_smiles.add(smiles)
                    candidates.append({
                        "smiles": smiles,
                        "round": batch_idx,
                        "proposer": "baseline",
                        "votes": {},
                        "avg_vote": 0
                    })
                    new_unique += 1

            logger.info(f"Batch {batch_idx}: Got {len(unique_smiles_list)} valid, {new_unique} new unique. "
                       f"Total: {len(collected_smiles)}/{config.num_molecules}")

            # Store message for debugging
            messages.append({
                "batch": batch_idx,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "response": response_text[:1000] + "..." if len(response_text) > 1000 else response_text,
                "parsed_count": len(unique_smiles_list),
                "new_unique": new_unique,
            })

            # Log progress to wandb
            if batch_idx % 5 == 0:
                wandb.log({
                    "batch": batch_idx,
                    "total_unique_smiles": len(collected_smiles),
                    "batch_valid_count": len(unique_smiles_list),
                    "batch_new_unique": new_unique,
                    "running_cost_usd": cb.total_cost,
                })

        # Trim to exact count if we overshot
        candidates = candidates[:config.num_molecules]

        logger.info(f"\nGeneration complete. Collected {len(candidates)} unique molecules.")
        logger.info(f"Total batches: {batch_idx}")
        logger.info(f"LLM cost: ${cb.total_cost:.4f} ({cb.total_tokens} tokens)")

        # Review candidates
        logger.info("\nReviewing candidates...")
        overall_score, total_detailed_results = review_candidates(
            model=None,  # Not used
            candidates=candidates,
            task_name=config.task_name,
            seed_mol_index=config.seed_mol_index,
            sim_threshold=config.sim_threshold,
            freq_log=config.freq_log,
            scientist_profiles={},  # Empty for baseline
            run_name=wandb.run.name if wandb.run else "baseline",
            num_candidates=config.num_molecules
        )
        wandb.log(overall_score, step=batch_idx)
        df = pd.DataFrame(total_detailed_results)
        wandb_table = wandb.Table(dataframe=df)
        wandb.log({"detailed_results": wandb_table}, step=batch_idx)
        # Log final metrics
        wandb.log({
            "total_cost_usd": cb.total_cost,
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_batches": batch_idx,
        })

        return {
            "candidates": total_detailed_results,
            "messages": messages,
            "total_cost_usd": cb.total_cost,
            "total_tokens": cb.total_tokens,
            "overall_score": overall_score,
            "config": {
                "task_name": config.task_name,
                "num_molecules": config.num_molecules,
                "batch_size": config.batch_size,
                "temperature": config.temperature,
                "seed_mol_index": config.seed_mol_index,
                "sim_threshold": config.sim_threshold,
            },
        }


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Baseline Molecular Generation (no debate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m src.agent.baseline --task_name lead_optimization/parp1
  uv run python -m src.agent.baseline --task_name pmo/GSK3B --num_candidates 500 --batch_size 50
        """
    )

    parser.add_argument("--task_name", default="lead_optimization/parp1",
                        help="Molecular optimization task name")
    parser.add_argument("--model", "-m", default="gpt-4o-mini",
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="LLM temperature (default: 0.7)")
    parser.add_argument("--seed_mol_index", type=int, default=1,
                        help="Seed molecule index for lead optimization task")
    parser.add_argument("--sim_threshold", type=float, default=0.4,
                        help="Similarity threshold for lead optimization task")
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        help="Wandb mode (online, offline, disabled)")
    parser.add_argument("--freq_log", type=int, default=100,
                        help="Frequency of logging (default: 100)")
    parser.add_argument("--num_candidates", type=int, default=1000,
                        help="Number of molecules to generate (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Molecules per LLM call (default: 20)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top k candidates to display (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of maximum rounds to generate molecules (default: 20)")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize wandb
    task_short = args.task_name.split('/')[-1]
    task_group = args.task_name.split('/')[0]
    run_name = f"{task_short}_{args.seed_mol_index}_{args.sim_threshold}_{args.model}_baseline"

    wandb.init(
        project="clever-hans",
        group=task_group,
        mode=args.wandb_mode,
        name=run_name
    )
    wandb.config.update(args)

    # Get task description
    task_description = get_task_description(args.task_name, args.seed_mol_index, args.sim_threshold)

    # Build config
    baseline_config = BaselineConfig(
        task_name=args.task_name,
        num_molecules=args.num_candidates,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed_mol_index=args.seed_mol_index,
        sim_threshold=args.sim_threshold,
        freq_log=args.freq_log,
        top_k=args.top_k,
        rounds=args.rounds,
    )

    # Run baseline generation
    results = run_baseline(
        task_description=task_description,
        model_name=args.model,
        config=baseline_config,
    )

    # Save output
    output_dir = f"output/{args.task_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = f"{output_dir}/baseline_{wandb.run.name}_{wandb.run.id}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Print top results
    candidates = results.get("candidates", [])
    if candidates:
        print("\n" + "=" * 60)
        print(f"TOP {args.top_k} MOLECULES")
        print("=" * 60)
        for mol in candidates[:args.top_k]:
            print(f"\nRank {mol.get('rank', '?')}: {mol['smiles']}")
            print(f"  Score: {mol.get('score', 0):.4f}")
            if mol.get('score_details'):
                details = mol['score_details']
                for k, v in details.items():
                    if v is not None:
                        print(f"  {k}: {v}")
        print("=" * 60)
    else:
        print("\nNo valid molecules generated.")

    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()
