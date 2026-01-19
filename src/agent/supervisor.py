# Supervisor agent for selecting scientists for the molecular optimization debate.

import logging
from typing import Dict, List, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from src.agent.tools import get_scientists

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """You are the Supervisor Agent for a molecular optimization project.

Your role is to identify the most relevant scientists for the given task by analyzing
their publication history and expertise in the relevant domain.

When given a task:
1. Use the get_scientists tool to search for relevant researchers
2. The tool will find publications relevant to the task and extract author names
3. Consider scientists who have expertise in:
   - The target protein/pathway mentioned
   - Similar molecular scaffolds
   - Relevant assay development or screening methods

Return the complete list of scientist names that will participate in the debate.
Be thorough in your search to ensure diverse expertise.
"""


def create_supervisor_agent(model):
    """Create the supervisor agent for scientist selection.

    Args:
        model: The LLM model to use

    Returns:
        Compiled ReAct agent for scientist selection
    """
    agent = create_agent(
        model=model,
        tools=[get_scientists],
        system_prompt=SUPERVISOR_PROMPT,
    )

    return agent


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


def _extract_scientists_from_result(result: Dict[str, Any]) -> List[str]:
    """Extract scientist names from agent result.

    Args:
        result: The agent's result dictionary

    Returns:
        List of scientist names
    """
    scientists = []

    messages = result.get("messages", [])
    for msg in messages:
        if msg.type == 'tool':
            # Check for tool messages with scientist lists
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                scientists.extend(msg.content)
            elif hasattr(msg, 'content') and isinstance(msg.content, str):
                # Try to parse list from string representation
                content = msg.content
                if '[' in content and ']' in content:
                    try:
                        import ast
                        # Find list in content
                        start = content.find('[')
                        end = content.rfind(']') + 1
                        list_str = content[start:end]
                        parsed = ast.literal_eval(list_str)
                        if isinstance(parsed, list):
                            scientists.extend(parsed)
                    except (ValueError, SyntaxError):
                        pass

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in scientists:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique
