import sqlite3
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PARQUET_PATH = DATA_DIR / "molecules_by_author_all.parquet"
SQLITE_PATH = DATA_DIR / "molecules_by_author.db"


def convert_parquet_to_sqlite(
    parquet_path: Path = PARQUET_PATH,
    sqlite_path: Path = SQLITE_PATH,
    chunk_size: int = 500_000
) -> None:
    """Convert parquet file to SQLite database with proper indexing.

    Args:
        parquet_path: Path to source parquet file
        sqlite_path: Path to destination SQLite database
        chunk_size: Number of rows to process at a time
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Remove existing database if it exists
    if sqlite_path.exists():
        sqlite_path.unlink()
        logger.info(f"Removed existing database: {sqlite_path}")

    logger.info(f"Converting {parquet_path} to {sqlite_path}...")

    # Create database and table
    conn = sqlite3.connect(str(sqlite_path))

    # Read parquet in chunks and insert into SQLite
    df_iter = pd.read_parquet(
        parquet_path,
        columns=['author', 'author_raw', 'smiles', 'inchi_key', 'title', 'pubmed_id', 'document_chembl_id', 'doi']
    )

    # Since read_parquet doesn't support chunked reading directly,
    # we read all and then insert in chunks
    logger.info("Reading parquet file...")
    df = pd.read_parquet(
        parquet_path,
        columns=['author', 'author_raw', 'smiles', 'inchi_key', 'title', 'pubmed_id', 'document_chembl_id', 'doi']
    )

    total_rows = len(df)
    logger.info(f"Total rows: {total_rows:,}")

    # Normalize author column for consistent matching
    df['author_lower'] = df['author'].str.lower().str.strip()

    # Insert in chunks
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk.to_sql(
            'molecules',
            conn,
            if_exists='append' if i > 0 else 'replace',
            index=False
        )
        logger.info(f"Inserted rows {i:,} to {min(i + chunk_size, total_rows):,}")

    # Create indexes for fast lookups
    logger.info("Creating indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_author_lower ON molecules(author_lower)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_smiles ON molecules(smiles)")

    conn.commit()
    conn.close()

    logger.info(f"Conversion complete. Database saved to {sqlite_path}")
    logger.info(f"Database size: {sqlite_path.stat().st_size / (1024**3):.2f} GB")