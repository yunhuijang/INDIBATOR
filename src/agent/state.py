"""State definitions for the multi-agent molecular optimization system."""

from dataclasses import dataclass
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph.message import add_messages


@dataclass
class DebateConfig:
    """Configuration for debate orchestration."""
    task_name: str
    max_rounds: int = 3
    consensus_threshold: float = 0.66
    is_summary_agent: bool = False
    top_k: int = 10
    seed_mol_index: int = 0
    sim_threshold: float = 0.4
    num_scientists: int = 2
    temperature: float = 0.7
    use_vanilla_scientist_agent: bool = False
    num_mols_per_scientist: int = 3
    freq_log: int = 100
    num_candidates: int = 1000
    is_self_critique_on: bool = False
    is_rag_keyword: bool = False
    min_rounds: int = 0
    is_critque_on: bool = True
    is_voting_on: bool = True
    is_molecule_profile_on: bool = True
    is_publication_profile_on: bool = True


class ScientistProfile(TypedDict):
    """Profile of a scientist with their expertise."""
    name: str
    publications: List[Dict[str, str]]  # [{title, abstract}]
    molecules: List[Dict[str, str]]  # [{smiles, inchi_key, title, pubmed_id}]


class MoleculeCandidate(TypedDict):
    """A molecule proposed during debate."""
    smiles: str
    proposer: str
    round: int
    critiques: List[str]
    votes: Dict[str, float]  # {scientist_name: vote_score}
    score: Optional[float]  # Reviewer score


class DebateMessage(TypedDict):
    """A message in the debate transcript."""
    speaker: str
    content: str
    round: int
    message_type: str  # "proposal", "critique", "vote"


class DebateState(TypedDict):
    """Full state for the debate workflow."""
    # Task input
    task_description: str

    # Scientist selection
    scientist_names: List[str]
    scientist_profiles: Dict[str, ScientistProfile]

    # Debate tracking
    messages: Annotated[List[DebateMessage], add_messages]
    candidates: List[MoleculeCandidate]
    current_round: int
    max_rounds: int
    current_speaker_idx: int
    phase: str  # "init", "proposal", "critique", "voting", "complete"

    # Final output
    final_output: Optional[List[Dict]]  # Scored and ranked molecules
