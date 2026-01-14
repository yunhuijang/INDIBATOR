"""Scientist agent factory for creating domain-expert agents."""

import logging
from typing import Dict, List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from src.agent.state import ScientistProfile
from src.db.queries import get_publications_by_author, get_molecules_by_author
from src.agent.summarizer import summarize_publications, summarize_molecules
from src.utils import extract_content, truncate_for_prompt

logger = logging.getLogger(__name__)

SCIENTIST_PROMPT_TEMPLATE = """You are {scientist_name}, a researcher specializing in molecular design and drug discovery.

Your expertise is based on your published work:
{publication_summary}

Molecules you have worked with:
{molecule_summary}

CURRENT TASK: {task_description}

In this debate, you will participate in three phases:

1. PROPOSAL PHASE: Propose 1-3 molecules (as valid SMILES strings) based on your expertise.
   - Draw from your knowledge of similar molecular scaffolds
   - Consider structure-activity relationships from your publications
   - Explain your rationale for each proposal

2. CRITIQUE PHASE: Evaluate other scientists' proposals.
   - Identify potential issues (toxicity, synthesis difficulty, selectivity)
   - Suggest modifications based on your experience
   - Note any overlap with molecules in your experience

3. VOTING PHASE: Score candidates from 0.0 to 1.0.
   - Consider task relevance, synthetic feasibility, and novelty
   - Base your assessment on your domain expertise

Always ground your contributions in your specific published expertise.
When proposing molecules, provide valid SMILES strings and clear scientific rationale.
"""

def create_vanilla_scientist_agent(model, task_description: str, index: int):
    """Create a vanilla scientist agent. (without any name and profile)"""
    
    prompt = SCIENTIST_PROMPT_TEMPLATE.format(
        scientist_name=f"Scientist {index}",
        publication_summary="",
        molecule_summary="",
        task_description=task_description
    )

    # No tools needed - the scientist's profile is already in the system prompt
    # Removing get_publications tool prevents redundant DB queries during debate phases
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=prompt,
    )

    return agent

def create_scientist_agent(model, scientist_name: str, profile: ScientistProfile, task: str, publication_summary_agent: CompiledStateGraph=None, molecule_summary_agent: CompiledStateGraph=None):
    """Create a scientist agent with loaded expertise.

    Args:
        model: The LLM model to use
        scientist_name: Name of the scientist
        profile: Scientist's profile with publications and molecules
        task: The molecular optimization task

    Returns:
        Compiled ReAct agent for this scientist
    """
    # Summarize publications (top 10)
    publications = profile.get('publications', [])
    if publications:
        if publication_summary_agent:
            pub_summary = summarize_publications(publication_summary_agent, task, publications)
            pub_summary = extract_content(pub_summary)
        else:
            pub_summary = "\n".join([
                f"- {p.get('title', 'Untitled')}"
                for p in publications
            ])
    else:
        pub_summary = "No publications loaded from database."

    # Summarize molecules (top 10)
    molecules = profile.get('molecules', [])
    if molecules:
        if molecule_summary_agent:
            mol_summary = summarize_molecules(molecule_summary_agent, task, molecules)
            mol_summary = extract_content(mol_summary)
        else:
            mol_summary = "\n".join([
                f"- {m.get('smiles', 'N/A')} (from: {m.get('title', 'Unknown publication')}...)"
                for m in molecules
            ])
    else:
        mol_summary = "No molecules in database for this author."

    # Format the prompt
    prompt = SCIENTIST_PROMPT_TEMPLATE.format(
        scientist_name=scientist_name,
        publication_summary=pub_summary,
        molecule_summary=mol_summary,
        task_description=task
    )

    # No tools needed - the scientist's profile is already in the system prompt
    # Removing get_publications tool prevents redundant DB queries during debate phases
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=prompt,
    )

    return agent


def load_scientist_profiles(scientist_names: List[str]) -> Dict[str, ScientistProfile]:
    """Load profiles for all selected scientists.

    Args:
        scientist_names: List of scientist names to load

    Returns:
        Dictionary mapping names to profiles
    """
    profiles = {}

    for name in scientist_names:
        publications = get_publications_by_author(name)
        molecules = get_molecules_by_author(name)

        profiles[name] = {
            "name": name,
            "publications": publications,
            "molecules": molecules
        }

        logger.info(
            f"Loaded profile for {name}: "
            f"{len(publications)} publications, {len(molecules)} molecules"
        )

    return profiles


def get_scientist_proposal(agent, task: str, round_num: int, previous_proposals: str, num_mols: int) -> str:
    """Get a proposal from a scientist agent.

    Args:
        agent: The scientist's agent
        task: The optimization task
        round_num: Current debate round
        previous_proposals: Summary of previous proposals

    Returns:
        The scientist's proposal response
    """
    # Truncate previous proposals to prevent token overflow
    previous_proposals = truncate_for_prompt(previous_proposals, max_chars=100000)

    prompt = f"""Round {round_num} - PROPOSAL PHASE

Task: {task}

Previous proposals in this debate:
{previous_proposals if previous_proposals else "No previous proposals yet."}

Based on your expertise, propose {num_mols}-{num_mols+2} novel molecules (as SMILES strings) that could address this task.
For each molecule:
1. Provide the SMILES string
2. Explain your rationale based on your published work
3. Discuss expected properties relevant to the task

Output format:
[
    {{"SMILES": "SMILES string", "rationale": "Brief rationale"}},
    ...
]

IMPORTANT:
- Output ONLY the JSON array, no other text or markdown
- Keep rationales brief (3-4 sentences) to avoid truncation
- Ensure all brackets and quotes are properly closed
- Do NOT wrap in code blocks
- Ensure the SMILES strings are valid and have not proposed in previous proposals
"""

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return extract_content(result)


def get_scientist_critique(agent, task: str, round_num: int, proposals_list: list) -> str:
    """Get critiques from a scientist agent.

    Args:
        agent: The scientist's agent
        task: The optimization task
        round_num: Current debate round
        proposals_to_critique: Other scientists' proposals

    Returns:
        The scientist's critique response
    """
    # Truncate proposals to prevent token overflow
    proposals_to_critique = truncate_for_prompt(proposals_list, max_chars=100000)

    prompt = f"""Round {round_num} - CRITIQUE PHASE

Task: {task}

Review these proposals from other scientists:
{proposals_to_critique}

Based on your expertise, critique each proposal:
- Identify potential issues (toxicity, synthesis difficulty, selectivity)
- Suggest specific modifications if appropriate
- Note any overlap with molecules in your experience
- Highlight promising aspects that align with your research

Output format:
[
    {{"SMILES": "SMILES string", "proposer": "Name", "critique": "Brief critique"}},
    ...
]

IMPORTANT:
- Output ONLY the JSON array, no other text or markdown
- Keep critiques brief (3-4 sentences) to avoid truncation
- Ensure all brackets and quotes are properly closed
- Do NOT wrap in code blocks
"""

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return extract_content(result)


def get_scientist_votes(agent, task: str, round_num: int, all_candidates: str) -> str:
    """Get votes from a scientist agent.

    Args:
        agent: The scientist's agent
        task: The optimization task
        round_num: Current debate round
        all_candidates: All proposed molecules

    Returns:
        The scientist's voting response
    """
    # Truncate candidates list to prevent token overflow
    all_candidates = truncate_for_prompt(all_candidates, max_chars=100000)

    prompt = f"""Round {round_num} - VOTING PHASE

Task: {task}

All candidate molecules proposed in this debate:
{all_candidates}

Vote for your top candidates by assigning scores from 0.0 to 1.0.
Format each vote as: SMILES: score

Consider:
- Relevance to the task (binding affinity to target)
- Synthetic feasibility
- Novelty compared to existing drugs
- Critiques received from other scientists

Provide scores for at least top 3 candidates with brief justifications.

Output format:
[
    {{"SMILES": "SMILES string", "score": 0.8, "justification": "Brief justification"}},
    ...
]

IMPORTANT:
- Output ONLY the JSON array, no other text or markdown
- Keep justifications brief (3-4 sentences) to avoid truncation
- Score must be a number between 0.0 and 1.0
- Ensure all brackets and quotes are properly closed
- Do NOT wrap in code blocks
"""

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return extract_content(result)