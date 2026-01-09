"""Debate orchestrator for multi-round molecular optimization debates."""

import logging
import re
from typing import Dict, List, Literal, Any
import json
import ast
import wandb
import pandas as pd
from langgraph.graph import StateGraph, END

from src.agent.state import DebateState, MoleculeCandidate, DebateMessage, DebateConfig
from src.agent.scientist import (
    create_scientist_agent,
    load_scientist_profiles,
    get_scientist_proposal,
    get_scientist_critique,
    get_scientist_votes,
)
from src.agent.summarizer import create_summarizer_agent
from src.agent.reviewer import review_candidates
from src.utils import canonicalize_smiles

logger = logging.getLogger(__name__)


class DebateOrchestrator:
    """Orchestrates multi-round debates between scientist agents."""

    def __init__(self, model, config: DebateConfig):
        """Initialize the debate orchestrator.

        Args:
            model: The LLM model to use for scientist agents
            config: Configuration for debate orchestration
        """
        self.model = model
        self.config = config
        self.max_rounds = config.max_rounds
        self.consensus_threshold = config.consensus_threshold
        self.is_summary_agent = config.is_summary_agent
        self.task_name = config.task_name
        self.seed_mol_index = config.seed_mol_index
        self.sim_threshold = config.sim_threshold
        self.num_candidates = config.num_candidates
        self.num_scientists = config.num_scientists
        self.scientist_agents = {}
        self.top_k = config.top_k
        
    def run_debate(self, task_description: str, scientist_names: List[str]) -> Dict[str, Any]:
        """Run a complete debate session.

        Args:
            task_description: The molecular optimization task
            scientist_names: List of scientists to participate

        Returns:
            Final state with candidates and debate history
        """
        logger.info(f"Starting debate with {len(scientist_names)} scientists")

        # Initialize state
        state = {
            "task_description": task_description,
            "scientist_names": scientist_names,
            "scientist_profiles": {},
            "messages": [],
            "candidates": [],
            "current_round": 1,
            "max_rounds": self.max_rounds,
            "current_speaker_idx": 0,
            "phase": "init",
            "final_output": None,
        }

        # Load scientist profiles
        state = self._initialize_debate(state)

        # Run debate rounds
        while state["current_round"] <= self.max_rounds:
            round_num = state["current_round"]
            logger.info(f"=== Round {round_num} ===")

            # Proposal phase
            state = self._proposal_phase(state)

            # Critique phase
            state = self._critique_phase(state)

            # Voting phase
            state = self._voting_phase(state)

            # Check convergence
            # if self._check_consensus(state) and len(state["candidates"]) >= self.num_candidates:

            
            candidates = state["candidates"]
            if len(candidates) > 0:
                overall_score, detailed_results = review_candidates(self.model, candidates, self.task_name, self.seed_mol_index, self.sim_threshold)
                wandb.log(overall_score, step=round_num)
                wandb_table = wandb.Table(dataframe=pd.DataFrame(detailed_results))
                wandb.log({"detailed_results": wandb_table}, step=round_num)
                
                # if self._check_consensus(state):
                #     logger.info("Consensus reached, ending debate early")
                #     break
                
                state["current_round"] += 1
            else:
                logger.info("No candidates to review, re-run the debate")
                state["current_round"] += 1

        # Aggregate results
        state = self._aggregate_results(state)

        return state

    def _initialize_debate(self, state: DebateState) -> DebateState:
        """Initialize debate with scientist profiles and agents."""
        logger.info("Loading scientist profiles...")

        profiles = load_scientist_profiles(state["scientist_names"])
        sorted_profiles = sorted(profiles.items(), key=lambda x: len(x[1]['molecules'])+len(x[1]['publications']), reverse=True)
        sorted_profiles = dict(sorted_profiles[:self.num_scientists])
        state["scientist_profiles"] = sorted_profiles
        if self.is_summary_agent:
            publication_summary_agent = create_summarizer_agent(self.model, 'publications',state['task_description'])
            molecule_summary_agent = create_summarizer_agent(self.model, 'molecules', state['task_description'])
        else:
            publication_summary_agent = None
            molecule_summary_agent = None
        # Create agents for each scientist
        for name, profile in sorted_profiles.items():
            self.scientist_agents[name] = create_scientist_agent(
                self.model, name, profile, state["task_description"], publication_summary_agent=publication_summary_agent, molecule_summary_agent=molecule_summary_agent
            )

        state["phase"] = "proposal"
        return state

    def _proposal_phase(self, state: DebateState) -> DebateState:
        """Run the proposal phase where scientists propose molecules."""
        round_num = state["current_round"]
        logger.info(f"Round {round_num}: Proposal phase")

        new_messages = []
        new_candidates = []

        # Format previous proposals for context
        previous = self._format_previous_proposals(state)

        for name in state["scientist_names"]:
            agent = self.scientist_agents.get(name)
            if not agent:
                continue

            response = get_scientist_proposal(
                agent, state["task_description"], round_num, previous
            )

            # Extract SMILES from response
            smiles_list = self._extract_smiles(response)
            logger.info(f"{name} proposed {len(smiles_list)} molecules")

            for smiles in smiles_list:
                new_candidates.append({
                    "smiles": smiles,
                    "proposer": name,
                    "round": round_num,
                    "critiques": [],
                    "votes": {},
                    "score": None,
                    "justifications": {},
                })

            new_messages.append({
                "speaker": name,
                "content": response,
                "round": round_num,
                "message_type": "proposal",
            })

        state["messages"] = state["messages"] + new_messages
        state["candidates"] = state["candidates"] + new_candidates
        state["phase"] = "critique"
        return state

    def _critique_phase(self, state: DebateState) -> DebateState:
        """Run the critique phase where scientists evaluate proposals."""
        round_num = state["current_round"]
        logger.info(f"Round {round_num}: Critique phase")

        new_messages = []

        # Get current round's candidates
        current_candidates = [
            c for c in state["candidates"] if c["round"] == round_num
        ]

        for name in state["scientist_names"]:
            agent = self.scientist_agents.get(name)
            if not agent:
                continue

            # Get proposals from OTHER scientists to critique
            others_proposals = [
                c for c in current_candidates if c["proposer"] != name
            ]

            if not others_proposals:
                continue

            proposals_list = self._format_proposals_for_critique(others_proposals)
            response = get_scientist_critique(
                agent, state["task_description"], round_num, proposals_list
            )
            response_dict = {r['SMILES']: {'proposer': r['proposer'], 'critique': r['critique']} for r in ast.literal_eval(response)}
            new_messages.append({
                "speaker": name,
                "content": response,
                "round": round_num,
                "message_type": "critique",
            })

            # Add critique to candidates
            for candidate in current_candidates:
                if candidate["proposer"] != name and candidate["smiles"] in response_dict:
                    candidate["critiques"].append(response_dict[candidate["smiles"]]['critique'])

        state["messages"] = state["messages"] + new_messages
        state["phase"] = "voting"
        return state

    def _voting_phase(self, state: DebateState) -> DebateState:
        """Run the voting phase where scientists score candidates."""
        round_num = state["current_round"]
        logger.info(f"Round {round_num}: Voting phase")

        new_messages = []
        all_candidates = state["candidates"]

        candidates_text = self._format_all_candidates(all_candidates)

        for name in state["scientist_names"]:
            agent = self.scientist_agents.get(name)
            if not agent:
                continue

            response = get_scientist_votes(
                agent, state["task_description"], round_num, candidates_text
            )

            # Parse votes from response
            votes = self._parse_votes(response)
            logger.info(f"{name} cast {len(votes)} votes")

            # Apply votes to candidates
            for smiles, vote_result in votes.items():
                for candidate in all_candidates:
                    if candidate["smiles"] == smiles:
                        candidate["votes"][name] = vote_result['score']
                        candidate["justifications"][name] = vote_result['justification']

            new_messages.append({
                "speaker": name,
                "content": response,
                "round": round_num,
                "message_type": "vote",
            })

        state["messages"] = state["messages"] + new_messages
        state["phase"] = "proposal"
        return state

    def _check_consensus(self, state: DebateState) -> bool:
        """Check if consensus has been reached."""
        candidates = state["candidates"]
        if not candidates:
            return False

        # Calculate average votes
        for c in candidates:
            if c["votes"]:
                scores = [float(score) for score in c["votes"].values()]
                c["avg_vote"] = sum(scores) / len(scores)
            else:
                c["avg_vote"] = 0

        # Check if top candidate exceeds threshold
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("avg_vote", 0), reverse=True
        )

        if sorted_candidates and sorted_candidates[0].get("avg_vote", 0) >= self.consensus_threshold:
            return True

        return False

    def _aggregate_results(self, state: DebateState) -> DebateState:
        """Aggregate final results from the debate."""
        logger.info("Aggregating debate results...")

        candidates = state["candidates"]

        # Calculate final scores
        for c in candidates:
            if c["votes"]:
                scores = [float(score) for score in c["votes"].values()]
                c["avg_vote"] = sum(scores) / len(scores)
            else:
                c["avg_vote"] = 0

        # Sort by average vote
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("avg_vote", 0), reverse=True
        )

        # Keep top 10
        state["candidates"] = sorted_candidates[:self.num_candidates]
        state["phase"] = "complete"

        logger.info(f"Debate complete. Top candidates: {len(state['candidates'])}")
        return state

    def _extract_smiles(self, text: str) -> List[str]:
        """Extract SMILES strings from text.

        Uses regex to find SMILES-like patterns.
        """
        # TODO: Need to be fixed
        
        # Pattern for SMILES: starts with atom, contains typical SMILES characters
        
        result = ast.literal_eval(text)
        smiles_list = [canonicalize_smiles(r['SMILES']) for r in result]
        filtered_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        return filtered_smiles_list

    def _format_previous_proposals(self, state: DebateState) -> str:
        """Format previous proposals for context."""
        proposals = [
            m for m in state.get("messages", [])
            if m["message_type"] == "proposal"
        ]

        if not proposals:
            return "No previous proposals."

        lines = []
        for p in proposals[-10:]:  # Last 10 proposals
            snippet = p["content"][:300] + "..." if len(p["content"]) > 300 else p["content"]
            lines.append(f"{p['speaker']} (round {p['round']}): {snippet}")

        return "\n\n".join(lines)

    def _format_proposals_for_critique(self, candidates: List[MoleculeCandidate]) -> str:
        """Format proposals for the critique phase."""
        lines = []
        for c in candidates:
            lines.append({c['smiles']: c['proposer']})
        return lines

    def _format_all_candidates(self, candidates: List[MoleculeCandidate]) -> str:
        """Format all candidates for voting."""
        lines = []
        for i, c in enumerate(candidates, 1):
            vote_info = ""
            if c["votes"]:
                avg = sum([float(score) for score in c["votes"].values()]) / len(c["votes"])
                vote_info = f", current avg vote: {avg:.2f}"
            lines.append(
                f"{i}. {c['smiles']} (by {c['proposer']}, round {c['round']}{vote_info})"
            )
        return "\n".join(lines)

    def _parse_votes(self, text: str) -> Dict[str, float]:
        """Parse votes from text response.
        """
        result = ast.literal_eval(text)
        votes = {r['SMILES']: {'score': r['score'], 'justification': r['justification']} for r in result}

        return votes


def build_debate_graph(orchestrator: DebateOrchestrator) -> StateGraph:
    """Build a LangGraph StateGraph for the debate process.

    This provides an alternative graph-based execution model.

    Args:
        orchestrator: The debate orchestrator instance

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(DebateState)

    # Define nodes
    workflow.add_node("initialize", lambda s: orchestrator._initialize_debate(s))
    workflow.add_node("proposal", lambda s: orchestrator._proposal_phase(s))
    workflow.add_node("critique", lambda s: orchestrator._critique_phase(s))
    workflow.add_node("voting", lambda s: orchestrator._voting_phase(s))
    workflow.add_node("aggregate", lambda s: orchestrator._aggregate_results(s))

    # Define routing
    def should_continue(state: DebateState) -> Literal["continue", "end"]:
        if state["current_round"] >= orchestrator.max_rounds:
            return "end"
        if orchestrator._check_consensus(state):
            return "end"
        return "continue"

    # Define edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "proposal")
    workflow.add_edge("proposal", "critique")
    workflow.add_edge("critique", "voting")
    workflow.add_conditional_edges(
        "voting",
        should_continue,
        {"continue": "proposal", "end": "aggregate"}
    )
    workflow.add_edge("aggregate", END)

    return workflow.compile()
