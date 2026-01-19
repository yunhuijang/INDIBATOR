# Database query functions for retrieving publications and molecules.

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.utils import canonicalize_smiles, get_related_molecules, normalize_author_name
from prompt.task_rag_keyword import TASK_RAG_KEYWORD

logger = logging.getLogger(__name__)

# Database paths relative to project root
DB_PATH = Path(__file__).parent.parent.parent / "pubmed-vectors" / "pubmed_data_org.db"
MOLECULES_DB_PATH = Path(__file__).parent.parent.parent / "data" / "molecules_by_author.db"

# Module-level connections for connection pooling
_publications_conn: Optional[sqlite3.Connection] = None
_molecules_conn: Optional[sqlite3.Connection] = None


def _get_publications_conn() -> sqlite3.Connection:
    """Get or create a reusable connection for publications DB."""
    global _publications_conn
    if _publications_conn is None:
        _publications_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _publications_conn.row_factory = sqlite3.Row
    return _publications_conn


def _get_molecules_conn() -> sqlite3.Connection:
    """Get or create a reusable connection for molecules DB."""
    global _molecules_conn
    if _molecules_conn is None:
        _molecules_conn = sqlite3.connect(str(MOLECULES_DB_PATH), check_same_thread=False)
        _molecules_conn.row_factory = sqlite3.Row
    return _molecules_conn


def count_keywords_in_publication(title: str, abstract: str, keywords: List[str]) -> int:
    """Count how many keywords appear in both title and abstract."""
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    count = 0
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in title_lower and keyword_lower in abstract_lower:
            count += 1
    return count


def get_publications_by_author(author_name: str, task_name: str, limit: int = 100, final_candidate_num: int = 50) -> List[Dict[str, str]]:
    """Query SQLite for publications by author.

    Args:
        author_name: Author name in normalized format (e.g., "smith j")
        task_name: Task name for keyword matching
        limit: Maximum number of publications to return
        final_candidate_num: Final number of candidates after sorting

    Returns:
        List of dictionaries with 'title' and 'abstract' keys, sorted by keyword relevance
    """
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return []
    keywords = TASK_RAG_KEYWORD.get(task_name, [])
    try:
        conn = _get_publications_conn()
        # Uses index on authors_clean_2 for fast lookups

        # Total publication by author            
        cursor = conn.execute(
            "SELECT title, abstract FROM articles_final WHERE author_raw = ? LIMIT ?",
            (author_name, limit)
        )
        results_total = [
            {"title": row["title"] or "", "abstract": row["abstract"] or ""}
            for row in cursor.fetchall()
        ]
        # Keyword-filtered publication by author
        if len(keywords) > 0:
            conditions = []
            params = [author_name]
            for keyword in keywords:
                pattern = f"%{keyword.lower()}%"
                conditions.extend(["lower(abstract) LIKE ?", "lower(title) LIKE ?"])
                params.extend([pattern, pattern])
            
            query = f"SELECT title, abstract FROM articles_final WHERE author_raw = ? AND ({' OR '.join(conditions)}) LIMIT ?"
            params.append(limit)
            cursor = conn.execute(query, tuple(params))
            results_keyword = [
                {"title": row["title"] or "", "abstract": row["abstract"] or ""}
                for row in cursor.fetchall()
            ]
            
        logger.info(f"Found {len(results_total)} total publications for '{author_name}'")
        logger.info(f"Found {len(results_keyword)} keyword-filtered publications for '{author_name}'")
        
        results = results_keyword + results_total

        # Use decorate-sort-undecorate pattern to avoid O(n log n) keyword computations
        if len(results) > 0:
            # Compute scores once (O(n))
            scored_results = [
                (count_keywords_in_publication(r["title"], r["abstract"], keywords), r)
                for r in results
            ]
            # Sort by pre-computed score (no redundant computations)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            # Log best score (already computed)
            if scored_results:
                logger.info(f"Best publication for '{author_name}': {scored_results[0][0]}")
            # Extract sorted results
            results = [r for _, r in scored_results[:final_candidate_num]]
        else:
            results = results[:final_candidate_num]

        return results
    except Exception as e:
        logger.error(f"Error querying publications for '{author_name}': {e}")
        return []


def compute_max_similarity_to_references(smiles: str, reference_fps: List) -> float:
    """Compute maximum Tanimoto similarity between a molecule and reference fingerprints."""
    if not reference_fps or not smiles:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfgen.GetFingerprint(mol)
        mean_sim = np.mean([DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in reference_fps])
        return mean_sim
    except Exception:
        return 0.0


def get_molecules_by_author(author_name: str, task_name, limit: int = 1000, final_candidate_num: int = 50) -> List[Dict[str, str]]:
    """Query SQLite for molecules by author.

    Args:
        author_name: Author name in normalized format (e.g., "smith j")
        task_name: Task name for similarity computation
        limit: Maximum number of molecules to return
        final_candidate_num: Final number of candidates after sorting

    Returns:
        List of dictionaries with molecule information, sorted by similarity to task-related molecules
    """
    author_name = normalize_author_name(author_name)
    if not MOLECULES_DB_PATH.exists():
        logger.error(f"Molecules database not found at {MOLECULES_DB_PATH}")
        return []
    task_related_molecules = get_related_molecules(task_name)

    # Precompute fingerprints for reference molecules
    reference_fps = []
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for ref_smiles in task_related_molecules:
        try:
            ref_mol = Chem.MolFromSmiles(ref_smiles)
            if ref_mol is not None:
                fp = mfgen.GetFingerprint(ref_mol)
                reference_fps.append(fp)
        except Exception:
            continue

    try:
        conn = _get_molecules_conn()
        # Use author_lower column which has an index (idx_author_lower)
        # Normalize input to lowercase for consistent matching
        author_lower = author_name.lower().strip()
        cursor = conn.execute(
            "SELECT DISTINCT smiles, inchi_key, title, pubmed_id FROM molecules WHERE author_lower = ? LIMIT ?",
            (author_lower, limit)
        )

        # Build results with fingerprints computed once (avoid double SMILES parsing)
        results_with_fp = []
        for row in cursor.fetchall():
            smiles = row["smiles"]
            canonical = canonicalize_smiles(smiles)
            # Compute fingerprint once during construction
            fp = None
            sim = 0.0
            if reference_fps and canonical:
                try:
                    mol = Chem.MolFromSmiles(canonical)
                    if mol:
                        fp = mfgen.GetFingerprint(mol)
                        sim = np.mean([DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in reference_fps])
                except Exception:
                    pass
            results_with_fp.append((sim, {
                "smiles": canonical,
                "inchi_key": row["inchi_key"],
                "title": row["title"],
                "pubmed_id": row["pubmed_id"]
            }))

        logger.info(f"Found {len(results_with_fp)} molecules for '{author_name}'")
        if len(results_with_fp) < 2:
            return [r for _, r in results_with_fp]

        # Sort by pre-computed similarity (no redundant fingerprint computations)
        if reference_fps:
            results_with_fp.sort(key=lambda x: x[0], reverse=True)
            if results_with_fp:
                logger.info(f"Best molecule for '{author_name}': {round(results_with_fp[0][0], 4)}")

        # Extract sorted results without the similarity score
        results = [r for _, r in results_with_fp[:final_candidate_num]]
        return results
    except Exception as e:
        logger.error(f"Error querying molecules for '{author_name}': {e}")
        return []
