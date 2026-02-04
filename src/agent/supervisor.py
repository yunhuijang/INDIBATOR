# Supervisor agent for selecting scientists for the molecular optimization debate.

import logging
from typing import List

from src.agent.tools import get_scientists

logger = logging.getLogger(__name__)


def run_supervisor(model, rag_prompt, num_scientists: int = 10) -> List[str]:
    """Run the supervisor to select scientists.

    Args:
        model: The LLM model to use (kept for API compatibility)
        rag_prompt: The RAG query - either a string or list of keywords
        num_scientists: Target number of scientists to select

    Returns:
        List of scientist names
    """
    # Convert keyword list to query string if needed
    if isinstance(rag_prompt, list):
        query = " ".join(rag_prompt)
    else:
        query = rag_prompt

    logger.info(f"Supervisor selecting scientists for: {query[:50]}...")

    # Call get_scientists directly with exact query (bypass LLM reformulation)
    scientist_names = get_scientists.invoke({"task": query, "k": num_scientists})

    logger.info(f"Supervisor selected {len(scientist_names)} scientists")
    return scientist_names