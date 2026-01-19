"""LangChain tools for the multi-agent molecular optimization system."""

import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

from src.utils import find_matches, normalize_author_name
from src.db.queries import get_publications_by_author, get_molecules_by_author

logger = logging.getLogger(__name__)


# @tool
# def get_pubmed_results(query: str, k: int = 3) -> List[Dict[str, Any]]:
#     """Get PubMed results for a given query.

#     Args:
#         query: Search query string
#         k: Number of results to return

#     Returns:
#         List of matching publications with metadata
#     """
#     matches = find_matches(query, k)

#     results = []
#     for match in matches:
#         results.append({
#             'pmid': match['pmid'],
#             'distance': match['distance'],
#             'title': match['title'],
#             'authors': match['authors'],
#             'publication_year': match['publication_year'],
#             'abstract': match['abstract']
#         })

#     return results


@tool
def get_scientists(task: str, k: int = 10) -> List[str]:
    """Select scientists relevant to a molecular optimization task.

    Uses RAG to find relevant publications and extracts first/last authors.

    Args:
        task: Natural language description of the task
        k: Number of papers to search (returns up to 2*k authors)

    Returns:
        List of scientist names (normalized format)
    """
    matches = find_matches(task, 2*k)

    scientists = set()  # Use set to avoid duplicates

    for match in matches:
        authors_raw = match.get('authors', '')
        if not authors_raw:
            continue

        # Split authors by comma
        author_list = [a.strip() for a in authors_raw.split(',') if a.strip()]

        if len(author_list) >= 1:
            # First author
            try:
                first_author = author_list[0]
                if first_author:
                    scientists.add(first_author)
            except Exception as e:
                logger.warning(f"Failed to parse first author: {e}")

        if len(author_list) >= 2:
            # Last author
            try:
                last_author = author_list[-1]
                if last_author:
                    scientists.add(last_author)
            except Exception as e:
                logger.warning(f"Failed to parse last author: {e}")


    logger.info(f"Selected {len(scientists)} scientists for task: {task[:50]}...")
    return list(scientists)


@tool
def get_publications(author: str) -> Dict[str, Any]:
    """Get publications and molecules for a scientist.

    Retrieves the scientist's publication history and molecular work
    from the PubMed and ChEMBL databases.

    Args:
        author: Scientist name in normalized format

    Returns:
        Dictionary with 'author', 'publications', 'molecules', and counts
    """
    publications = get_publications_by_author(author)
    molecules = get_molecules_by_author(author)
    return {
        "author": author,
        "publications": publications,
        "molecules": molecules,
        "publication_count": len(publications),
        "molecule_count": len(molecules)
    }

