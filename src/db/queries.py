# Database query functions for retrieving publications and molecules.

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional

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


def get_publications_by_author(author_name: str, limit: int = 50) -> List[Dict[str, str]]:
    """Query SQLite for publications by author.

    Args:
        author_name: Author name in normalized format (e.g., "smith j")
        limit: Maximum number of publications to return

    Returns:
        List of dictionaries with 'title' and 'abstract' keys
    """
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return []

    try:
        conn = _get_publications_conn()
        # Uses index on authors_clean_2 for fast lookups
        cursor = conn.execute(
            "SELECT title, abstract FROM articles_final WHERE authors_clean_2 = ? LIMIT ?",
            (author_name, limit)
        )
        results = [
            {"title": row["title"] or "", "abstract": row["abstract"] or ""}
            for row in cursor.fetchall()
        ]
        logger.info(f"Found {len(results)} publications for '{author_name}'")
        return results
    except Exception as e:
        logger.error(f"Error querying publications for '{author_name}': {e}")
        return []


def get_molecules_by_author(author_name: str, limit: int = 50) -> List[Dict[str, str]]:
    """Query SQLite for molecules by author.

    Args:
        author_name: Author name in normalized format (e.g., "smith j")
        limit: Maximum number of molecules to return

    Returns:
        List of dictionaries with molecule information
    """
    if not MOLECULES_DB_PATH.exists():
        logger.error(f"Molecules database not found at {MOLECULES_DB_PATH}")
        return []

    try:
        conn = _get_molecules_conn()
        # Use author_lower column which has an index (idx_author_lower)
        # Normalize input to lowercase for consistent matching
        author_lower = author_name.lower().strip()
        cursor = conn.execute(
            "SELECT DISTINCT smiles, inchi_key, title, pubmed_id FROM molecules WHERE author_lower = ? LIMIT ?",
            (author_lower, limit)
        )
        results = [
            {
                "smiles": row["smiles"],
                "inchi_key": row["inchi_key"],
                "title": row["title"],
                "pubmed_id": row["pubmed_id"]
            }
            for row in cursor.fetchall()
        ]
        logger.info(f"Found {len(results)} molecules for '{author_name}'")
        return results
    except Exception as e:
        logger.error(f"Error querying molecules for '{author_name}': {e}")
        return []
