"""LangChain tools for the multi-agent molecular optimization system."""

import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

from src.utils import find_matches, normalize_author_name
from src.db.queries import get_publications_by_author, get_molecules_by_author

logger = logging.getLogger(__name__)


@tool
def get_pubmed_results(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Get PubMed results for a given query.

    Args:
        query: Search query string
        k: Number of results to return

    Returns:
        List of matching publications with metadata
    """
    matches = find_matches(query, k)

    results = []
    for match in matches:
        results.append({
            'pmid': match['pmid'],
            'distance': match['distance'],
            'title': match['title'],
            'authors': match['authors'],
            'publication_year': match['publication_year'],
            'abstract': match['abstract']
        })

    return results


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
                first_author = normalize_author_name(author_list[0])
                if first_author:
                    scientists.add(first_author)
            except Exception as e:
                logger.warning(f"Failed to parse first author: {e}")

        if len(author_list) >= 2:
            # Last author
            try:
                last_author = normalize_author_name(author_list[-1])
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


@tool
def compute_molecule_score(smiles: str, task_description: str) -> Dict[str, Any]:
    """Compute quality score for a molecule candidate.

    Placeholder function that returns mock scores.
    To be replaced with actual scoring logic (RDKit, docking, ADMET, etc.)

    Args:
        smiles: SMILES representation of the molecule
        task_description: The optimization task for context

    Returns:
        Dictionary with overall score and component metrics
    """
    import hashlib

    # Generate deterministic pseudo-random score based on SMILES hash
    # This ensures consistent scores for the same molecule
    hash_val = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
    base_score = (hash_val % 1000) / 1000  # 0.0 to 1.0

    # Adjust based on basic molecular properties
    score_adjustments = 0.0

    # Prefer medium-length molecules (not too simple, not too complex)
    mol_length = len(smiles)
    if 20 <= mol_length <= 100:
        score_adjustments += 0.1
    elif mol_length > 150:
        score_adjustments -= 0.1

    # Prefer molecules with nitrogen (often important for binding)
    if 'N' in smiles or 'n' in smiles:
        score_adjustments += 0.05

    # Prefer molecules with rings (aromatic systems)
    if 'c' in smiles or 'C1' in smiles:
        score_adjustments += 0.05

    overall_score = min(1.0, max(0.0, base_score * 0.7 + 0.3 + score_adjustments))

    return {
        "smiles": smiles,
        "overall_score": round(overall_score, 3),
        "validity": 1.0,  # Placeholder - would use RDKit to validate
        "druglikeness": round(base_score * 0.9 + 0.1, 3),  # Placeholder
        "synthetic_accessibility": round(base_score * 0.8 + 0.2, 3),  # Placeholder
        "predicted_binding": round(overall_score * 0.85, 3),  # Placeholder
    }
