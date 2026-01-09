# Summarizer agent for summarizing publications and molecules.

import logging
from typing import Dict, List, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# summary_type: "publications" or "molecules"
SUMMARIZER_PROMPT = """You are the Summarizer Agent for a molecular optimization project.

Your role is to summarize the {summary_type} for the given task.

When given a task:
1. Summarize the {summary_type} to something that can be used to help the scientist agents propose molecules for {description}.
2. Return the summary
"""

def create_summarizer_agent(model, summary_type: str, description: str):
    """Create a summarizer agent for summarizing publications and molecules.

    Args:
        model: The LLM model to use
        summary_type: The type of summary to create ("publications" or "molecules")
        description: The description of the task
    """
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SUMMARIZER_PROMPT.format(summary_type=summary_type, description=description)
    )
        
    return agent

def summarize_publications(agent, task_description: str, publications: List[Dict[str, Any]]):
    """Summarize the publications for the given task.

    Args:
        model: The LLM model to use
        description: The description of the task
    """
    prompt = f"""
    
    Task: {task_description}

    Publications: {publications}
    
    Summarize the publications that are relevant to the given task.
    Write down the summary in a way that can be used to help the scientist agents propose molecules for the given task.
    """
    
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return result

def summarize_molecules(agent, task_description: str, molecules: List[str]):
    """Summarize the molecules for the given task.

    Args:
        model: The LLM model to use
        description: The description of the task
    """
    prompt = f"""
    Task: {task_description}

    Molecules: {molecules}

    Summarize the molecules that are relevant to the given task.
    If there exists a common structural characterstics (e.g., functional group, ring structure, etc.) among the molecules, mention it.
    Write down the summary in a way that can be used to help the scientist agents propose molecules for the given task.
    """
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return result