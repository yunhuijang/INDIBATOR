"""Debate orchestrator for multi-round molecular optimization debates."""

import logging
from typing import Dict, List, Literal, Any
import wandb
import pandas as pd
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI


from src.agent.state import DebateState, MoleculeCandidate, DebateConfig
from src.agent.scientist import create_scientist_agent, load_scientist_profiles, \
    get_scientist_proposal, get_scientist_critique, get_scientist_votes, \
    get_scientist_self_critique
from src.agent.summarizer import create_summarizer_agent
from src.agent.reviewer import review_candidates
from src.utils import canonicalize_smiles, safe_parse_json_list, extract_smiles
from src.evaluate.tools_evaluate import evaluate_lead, evaluate_pmo

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
        self.num_scientists = config.num_scientists
        self.scientist_agents = {}
        self.top_k = config.top_k
        self.num_mols_per_scientist = config.num_mols_per_scientist
        self.num_candidates = config.num_candidates
        self.freq_log = config.freq_log
        self.is_self_critique_on = config.is_self_critique_on
        self.min_rounds = config.min_rounds

    def run_debate(self, task_description: str, scientist_names: List[str], cb) -> Dict[str, Any]:
        """Run a complete debate session.

        Args:
            task_description: The molecular optimization task
            scientist_names: List of scientists to participate
            cb: OpenAI callback for cost tracking

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
        check_stop = lambda s: s["current_round"] < self.min_rounds or (s["current_round"] <= self.max_rounds and len(s['candidates']) < self.num_candidates)
        while check_stop(state):
            round_num = state["current_round"]
            logger.info(f"=== Round {round_num} ===")

            # Proposal phase
            state = self._proposal_phase(state)
            if self.task_name.startswith('lead_optimization'):
                state = self._update_scores(state)
            # Self critique phase (optional)
            if self.is_self_critique_on:
                state = self._self_critique_phase(state)
                if self.task_name.startswith('lead_optimization'):
                    state = self._update_scores(state)

            # Critique phase
            state = self._critique_phase(state)

            # Voting phase
            state = self._voting_phase(state)

            # Check convergence
            # Aggregate candidates by unique SMILES
            candidates = state["candidates"]
            smiles_to_candidates = {}

            for c in candidates:
                smiles = c["smiles"]
                if c["votes"]:
                    scores = [float(score) for score in c["votes"].values()]
                    c["avg_vote"] = sum(scores) / len(scores)
                else:
                    c["avg_vote"] = 0

                if smiles not in smiles_to_candidates:
                    # First occurrence - initialize aggregated entry
                    smiles_to_candidates[smiles] = {
                        "smiles": smiles,
                        "proposer": c["proposer"],  # Keep first proposer for compatibility
                        "proposers": [c["proposer"]],  # Track all proposers
                        "round": c["round"],  # Keep first round for compatibility
                        "rounds": [c["round"]],  # Track all rounds
                        "votes": dict(c["votes"]),
                        "avg_vote": c["avg_vote"],
                        "critiques": list(c["critiques"]),
                        "justifications": dict(c.get("justifications", {})),
                        "score": c.get("score"),
                    }
                else:
                    # Duplicate - aggregate votes and track additional proposers
                    existing = smiles_to_candidates[smiles]
                    if c["proposer"] not in existing["proposers"]:
                        existing["proposers"].append(c["proposer"])
                    if c["round"] not in existing["rounds"]:
                        existing["rounds"].append(c["round"])
                    # Merge votes (later votes override if same scientist voted again)
                    existing["votes"].update(c["votes"])
                    existing["critiques"].extend(c["critiques"])
                    existing["justifications"].update(c.get("justifications", {}))
                    # Recalculate avg_vote after merging
                    if existing["votes"]:

                        scores = [float(score) for score in existing["votes"].values()]
                        existing["avg_vote"] = sum(scores) / len(scores)

            # Sort unique candidates by total votes received
            unique_candidates = list(smiles_to_candidates.values())
            sorted_candidates = sorted(unique_candidates, key=lambda x: x.get("score", 0), reverse=True)
            selected_candidates = sorted_candidates
            if len(selected_candidates) > self.num_candidates:
                selected_candidates = sorted_candidates[:self.num_candidates]
            selected_candidates_smiles = [c['smiles'] for c in selected_candidates]
            for c in sorted_candidates:
                if c['smiles'] in selected_candidates_smiles:
                    c['is_selected'] = True
                else:
                    c['is_selected'] = False
            state['candidates'] = sorted_candidates


            if len(selected_candidates) > 0:
                total_overall_score, total_detailed_results = review_candidates(self.model, selected_candidates, self.task_name, self.seed_mol_index, self.sim_threshold, self.freq_log, state['scientist_profiles'], wandb.run.name, self.num_candidates)
                # for all candidates, update the score and detailed results
                wandb.log(total_overall_score, step=round_num)
                df = pd.DataFrame(total_detailed_results)
                wandb_table = wandb.Table(dataframe=df)
                wandb.log({"detailed_results": wandb_table}, step=round_num)
                wandb.log({
                    "total_cost_usd": cb.total_cost,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                }, step=round_num)

                if state["current_round"] == self.max_rounds:
                    logger.info("Max rounds reached, ending debate")
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
        profiles = load_scientist_profiles(state["scientist_names"], self.task_name)
        sorted_profiles = sorted(profiles.items(), key=lambda x: len(x[1]['molecules'])+len(x[1]['publications']), reverse=True)
        sorted_profiles = sorted_profiles[:self.num_scientists]
        sorted_profiles = dict(sorted_profiles)

        state["scientist_profiles"] = sorted_profiles
        state["scientist_names"] = list(sorted_profiles.keys())
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

        # Format all candidates for debate
        previous = self._format_previous_best_candidates_for_debate(state["candidates"])

        for name in state["scientist_names"]:
            agent = self.scientist_agents.get(name)
            if not agent:
                continue

            response = get_scientist_proposal(
                agent, state["task_description"], round_num, previous, self.num_mols_per_scientist
            )
            if isinstance(self.model, ChatGoogleGenerativeAI):
                parsed = safe_parse_json_list(response)
                response = parsed[0]['text'] if parsed and isinstance(parsed[0], dict) and 'text' in parsed[0] else response

            # Extract SMILES from response
            smiles_list = extract_smiles(response)
            unique_smiles_list = list(set(smiles_list))
            logger.info(f"{name} proposed {len(smiles_list)} molecules; {len(unique_smiles_list)} unique molecules")

            for smiles in unique_smiles_list:
                new_candidates.append({
                    "smiles": smiles,
                    "proposer": name,
                    "round": round_num,
                    "critiques": [],
                    "votes": {},
                    "score": 0,
                    "justifications": {},
                    "is_selected": False,
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
        logger.info(f"Round {round_num}: Proposal phase complete. {len(state['candidates'])} candidates proposed")
        return state

    def _self_critique_phase(self, state: DebateState) -> DebateState:
        """Run the self critique phase where scientists critique their own proposals and re-propose."""
        round_num = state["current_round"]
        logger.info(f"Round {round_num}: Self critique phase")

        new_messages = []
        new_candidates = []
        current_candidate_smiles = [c['smiles'] for c in state['candidates']]

        for name in state["scientist_names"]:
            agent = self.scientist_agents.get(name)
            if not agent:
                continue

            # Get this scientist's OWN proposals from current round
            own_proposals = [
                c for c in state["candidates"]
                if c["proposer"] == name and c["round"] == round_num
            ]

            if not own_proposals:
                continue

            # Format own proposals with scores
            own_proposals_with_scores = self._format_proposals_for_critique(own_proposals)

            # Get self-critique and improved proposals
            response = get_scientist_self_critique(
                agent, state["task_description"], round_num,
                own_proposals_with_scores, self.num_mols_per_scientist
            )

            if isinstance(self.model, ChatGoogleGenerativeAI):
                parsed = safe_parse_json_list(response)
                response = parsed[0]['text'] if parsed and isinstance(parsed[0], dict) and 'text' in parsed[0] else response

            # Extract improved SMILES from response
            smiles_list = list(set(extract_smiles(response)))
            original_smiles_list = list(set(self._extract_original_smiles(response)))
            logger.info(f"{name} re-proposed {len(smiles_list)} improved molecules after self-critique")


            for original_smiles in original_smiles_list:
                # Remove the original candidate from the list of candidates
                if original_smiles in current_candidate_smiles:
                    current_candidate = [c for c in state['candidates'] if c['smiles'] == original_smiles and c['proposer'] == name and c['round'] == round_num]
                    if len(current_candidate) > 0:
                        state['candidates'].remove(current_candidate[0])

            for smiles in smiles_list:
                new_candidates.append({
                    "smiles": smiles,
                    "proposer": name,
                    "round": round_num,
                    "critiques": [],
                    "votes": {},
                    "score": 0,
                    "justifications": {},
                    "is_selected": False,
                })

            new_messages.append({
                "speaker": name,
                "content": response,
                "round": round_num,
                "message_type": "self_critique",
            })

        state["messages"] = state["messages"] + new_messages
        state["candidates"] = state["candidates"] + new_candidates
        logger.info(f"Round {round_num}: Self critique phase complete. {len(state['candidates'])} candidates re-proposed")
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
            if isinstance(self.model, ChatGoogleGenerativeAI):
                parsed = safe_parse_json_list(response)
                response = parsed[0]['text'] if parsed and isinstance(parsed[0], dict) and 'text' in parsed[0] else response
            parsed_response = safe_parse_json_list(response)
            response_dict = {}
            for r in parsed_response:
                if isinstance(r, dict) and 'SMILES' in r:
                    response_dict[r['SMILES']] = {
                        'proposer': r.get('proposer', ''),
                        'critique': r.get('critique', '')
                    }
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
            if isinstance(self.model, ChatGoogleGenerativeAI):
                parsed = safe_parse_json_list(response)
                response = parsed[0]['text'] if parsed and isinstance(parsed[0], dict) and 'text' in parsed[0] else response
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
            candidates, key=lambda x: len(x.get("votes", {})), reverse=True
        )

        # Keep top 10
        state["candidates"] = sorted_candidates
        state["phase"] = "complete"

        logger.info(f"Debate complete. Top candidates: {len(state['candidates'])}")
        return state

    def _update_scores(self, state: DebateState) -> DebateState:
        """Update scores for candidates."""
        logger.info("Updating scores for candidates...")
        candidates = state['candidates']
        smiles_list = [c['smiles'] for c in candidates]
        _, result_df = evaluate_lead(smiles_list, self.task_name, self.seed_mol_index, self.sim_threshold)
        result_dict = result_df.set_index('smiles').to_dict(orient='index')
        for c in candidates:
            if c['smiles'] in result_dict:
                c['score'] = result_dict[c['smiles']]['ds']
                c['score_details'] = {'qed': result_dict[c['smiles']]['qed'], 'sa': result_dict[c['smiles']]['sa'], 'sim': result_dict[c['smiles']]['sim']}
            else:
                c['score'] = 0
        return state


    def _extract_original_smiles(self, text: str) -> List[str]:
        """Extract original SMILES strings from text.

        Uses safe parsing to handle malformed/truncated LLM responses.
        """
        result = safe_parse_json_list(text)
        smiles_list = []
        for r in result:
            if isinstance(r, dict) and 'original_SMILES' in r:
                canonical = canonicalize_smiles(r['original_SMILES'])
                if canonical:
                    smiles_list.append(canonical)
        return smiles_list

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
            snippet = p["content"][:1000] + "..." if len(p["content"]) > 1000 else p["content"]
            lines.append(f"{p['speaker']} (round {p['round']}): {snippet}")

        return "\n\n".join(lines)

    def _format_proposals_for_critique(self, candidates: List[MoleculeCandidate]) -> str:
        """Format proposals for the critique phase with scores."""

        lines = []

        # Get SMILES list for batch scoring
        smiles_list = [c['smiles'] for c in candidates]

        if not smiles_list:
            return lines

        # Lead optimization tasks
        if self.task_name.startswith('lead_optimization'):
            for c in candidates:
                lines.append({c['smiles']: {'proposer': c['proposer'], 'score': c['score'], 'score_details': {k: round(v, 3) for k, v in c['score_details'].items()}}})

        # PMO tasks
        else:
            for c in candidates:
                lines.append({c['smiles']: {'proposer': c['proposer']}})


        return lines

    def _format_all_candidates(self, candidates: List[MoleculeCandidate]) -> str:
        """Format all candidates for voting, with truncation."""
        lines = []
        # Limit to 50 candidates to prevent token overflow
        for i, c in enumerate(candidates, 1):
            vote_info = ""
            if c["votes"]:
                avg = sum([float(score) for score in c["votes"].values()]) / len(c["votes"])
                vote_info = f", current avg vote: {avg:.2f}"
            lines.append(
                f"{i}. {c['smiles']} (by {c['proposer']}, round {c['round']}{vote_info})"
            )
        result = "\n".join(lines)

        return result

    def _format_previous_best_candidates_for_debate(self, candidates: List[MoleculeCandidate]) -> str:
        """Format top 20 candidates (by score) for debate with size limits."""
        lines = []
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        for i, c in enumerate(sorted_candidates[:20], 1):
            vote_info = ""
            if c["votes"]:
                avg = sum([float(score) for score in c["votes"].values()]) / len(c["votes"])
                vote_info = f", current avg vote: {avg:.2f}"
            # Limit critiques and justifications more aggressively to prevent token overflow
            critiques_str = "\n".join([crit[:400] + '...' if len(crit) > 400 else crit for crit in c['critiques'][:5]])
            justifications_str = "\n".join([j[:400] + '...' if len(j) > 400 else j for j in list(c['justifications'].values())[:5]])

            current_text = f"{i}. {c['smiles']} (by {c['proposer']}, round {c['round']}{vote_info}):\n"
            current_text += f"critiques: {critiques_str}\n"
            current_text += f"justifications: {justifications_str}\n"
            current_text += f"score: {c['score']}\n"

            if 'score_details' in c:
                current_text += f"score_details (constraint): {c['score_details']}\n"

            lines.append(
                current_text
            )
        return "\n".join(lines)

    def _parse_votes(self, text: str) -> Dict[str, float]:
        """Parse votes from text response.

        Uses safe parsing to handle malformed/truncated LLM responses.
        """
        result = safe_parse_json_list(text)
        votes = {}
        for r in result:
            if isinstance(r, dict) and 'SMILES' in r:
                votes[r['SMILES']] = {
                    'score': r.get('score', 0),
                    'justification': r.get('justification', '')
                }
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
