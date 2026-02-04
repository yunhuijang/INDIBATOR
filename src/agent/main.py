"""Main entry point for the multi-agent molecular optimization system.

This module orchestrates the complete workflow:
1. Supervisor selects relevant scientists
2. Scientists debate molecular candidates
3. Reviewer scores the candidates
4. Final ranked results are returned
"""
import os
import torch

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")


import argparse
import logging
import sys
from typing import List, Dict, Any
import json
import wandb

import config  # Load .env file before initializing OpenAI client

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import get_openai_callback
from src.agent.supervisor import run_supervisor
from src.agent.debate import DebateOrchestrator
from src.agent.state import DebateConfig
from src.utils import get_task_description
from langchain_deepseek import ChatDeepSeek

from prompt.task_rag_keyword import TASK_RAG_KEYWORD




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_optimization(
    task_description: str,
    model_name: str,
    config: DebateConfig,
) -> List[Dict[str, Any]]:
    """Run the complete molecular optimization workflow.

    Args:
        task_description: Natural language description of the optimization task
        model_name: OpenAI model to use (default: gpt-4o-mini)
        config: Configuration for debate orchestration

    Returns:
        List of ranked molecules with scores and metadata
    """
    logger.info("=" * 60)
    logger.info("MULTI-AGENT MOLECULAR OPTIMIZATION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Task: {task_description}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Scientists: {config.num_scientists}")
    logger.info(f"Max rounds: {config.max_rounds}")
    logger.info(f"Is summary agent: {config.is_summary_agent}")
    logger.info(f"Is self-critique on: {config.is_self_critique_on}")
    logger.info(f"Top k: {config.top_k}")
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

    # Track LLM API usage cost
    with get_openai_callback() as cb:
        # Step 1: Supervisor selects scientists
        logger.info("\n[STEP 1] Supervisor selecting scientists...")

        if config.is_rag_keyword:
            rag_prompt = TASK_RAG_KEYWORD[config.task_name]
        else:
            rag_prompt = task_description
        scientist_names = run_supervisor(model, rag_prompt, config.num_scientists)

        if not scientist_names:
            logger.error("No scientists selected. Check RAG service connection.")
            logger.info(f"LLM cost before failure: ${cb.total_cost:.4f} ({cb.total_tokens} tokens)")
            return []

        logger.info(f"Selected scientists: {scientist_names}")

        # Step 2: Run debate
        logger.info("\n[STEP 2] Running debate...")
        orchestrator = DebateOrchestrator(model=model, config=config)
        debate_result = orchestrator.run_debate(task_description, scientist_names, cb)

        # Add cost info to debate_result and log to wandb
        debate_result["total_cost_usd"] = cb.total_cost
        debate_result["total_tokens"] = cb.total_tokens
        wandb.log({
            "total_cost_usd": cb.total_cost,
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
        })
        logger.info(f"Total LLM cost: ${cb.total_cost:.4f} ({cb.total_tokens} tokens)")

    if not os.path.exists(f"output/{config.task_name}"):
        os.makedirs(f"output/{config.task_name}")
    with open(f"output/{config.task_name}/{wandb.run.name}_{wandb.run.id}.json", "w") as f:
        json.dump(debate_result, f)
    candidates = debate_result.get("candidates", [])

    if not candidates:
        logger.warning("No candidates emerged from debate.")
        return []

    logger.info(f"Debate produced {len(candidates)} candidates")

    return debate_result


def print_results(results: List[Dict[str, Any]], top_k: int = 5):
    """Print formatted results to console.

    Args:
        results: List of scored molecules
        top_k: Number of top results to display
    """
    print("\n" + "=" * 60)
    print("FINAL RESULTS - TOP MOLECULES")
    print("=" * 60)

    for mol in results[:top_k]:
        print(f"\nRank {mol['rank']}: {mol['smiles']}")
        print(f"  Score: {mol['score']:.3f}")
        print(f"  Proposed by: {mol['proposer']} (round {mol['round']})")
        print(f"  Debate consensus: {mol['avg_vote']:.2f} ({len(mol['votes'])} votes)")

        if mol.get('score_details'):
            details = mol['score_details']
            print(f"  QED: {details.get('qed', 'N/A')}")
            print(f"  SA: {details.get('sa', 'N/A')}")
            print(f"  Sim: {details.get('sim', 'N/A')}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Molecular Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m src.agent.main "Design a molecule with high GSK3B binding affinity"
  uv run python -m src.agent.main --model gpt-4o --scientists 6 "Optimize for EGFR inhibition"
        """
    )

    parser.add_argument("--task_name", default="lead_optimization/parp1",
                        help="Molecular optimization task name")
    parser.add_argument("--model", "-m", default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--scientists", "-s", type=int,default=3,
        help="Number of scientists in debate (default: 3)")
    parser.add_argument("--rounds", "-r", type=int, default=3,
        help="Maximum debate rounds (default: 3)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
        help="LLM temperature (default: 0.7)")
    parser.add_argument("--top-k", "-k", type=int, default=5,
        help="Number of top results to display (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Enable verbose logging")
    parser.add_argument("--is_summary_agent", action="store_true", help="Enable summary agent")
    parser.add_argument("--top_k_candidates", type=int, default=10,
        help="Number of top candidates to keep (default: 10)")
    parser.add_argument("--consensus_threshold", type=float, default=0.66,
        help="Consensus threshold for debate (default: 0.66)")
    parser.add_argument("--seed_mol_index", type=int, default=1,
                        help="Seed molecule index for lead optimization task")
    parser.add_argument("--sim_threshold", type=float, default=0.4,
                        help="Similarity threshold for lead optimization task")
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        help="Wandb mode (online, offline, disabled)")
    parser.add_argument("--num_mols_per_scientist", type=int, default=3,
                        help="Number of molecules to propose per scientist (default: 3)")
    parser.add_argument("--freq_log", type=int, default=100,
                        help="Frequency of logging (default: 100)")
    parser.add_argument("--num_candidates", type=int, default=1000,
                        help="Number of candidates to keep (default: 1000)")
    parser.add_argument("--is_self_critique_on", action="store_true",
                        help="Enable self-critique phase where scientists critique and improve their own proposals")
    parser.add_argument("--is_rag_keyword", action="store_true", help="Enable keyword-based RAG")
    parser.add_argument("--min_rounds", type=int, default=0,
                        help="Minimum rounds to run (default: 0)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize wandb
    run_name = args.task_name.split('/')[1] + "_" + str(args.seed_mol_index) + "_" + str(args.sim_threshold) + "_" + args.model
    if args.is_self_critique_on:
        run_name += "_self_critique"

    wandb.init(
        project="clever-hans",
        group=args.task_name.split('/')[0],
        mode=args.wandb_mode,
        name=run_name
    )
    wandb.config.update(args)

    task_description = get_task_description(args.task_name, args.seed_mol_index, args.sim_threshold)
    config = DebateConfig(
        task_name=args.task_name,
        max_rounds=args.rounds,
        consensus_threshold=args.consensus_threshold,
        is_summary_agent=args.is_summary_agent,
        top_k=args.top_k_candidates,
        seed_mol_index=args.seed_mol_index,
        sim_threshold=args.sim_threshold,
        num_scientists=args.scientists,
        temperature=args.temperature,
        freq_log=args.freq_log,
        num_candidates=args.num_candidates,
        num_mols_per_scientist=args.num_mols_per_scientist,
        is_self_critique_on=args.is_self_critique_on,
        is_rag_keyword=args.is_rag_keyword,
        min_rounds=args.min_rounds,
    )
    # Run optimization
    results = run_optimization(
        task_description=task_description,
        model_name=args.model,
        config=config
    )

    # Sort results
    results_print = results['candidates']
    results_print.sort(key=lambda x: x.get('score', 0), reverse=True)
    for i, candidate in enumerate(results_print):
        candidate['rank'] = i + 1

    # Print results
    if results:
        print_results(results_print, top_k=args.top_k)
    else:
        print("\nNo results generated. Please check the logs for errors.")
        wandb.finish()

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    main()
