"""Normalize the articles table by splitting comma-separated authors into individual rows."""

import sqlite3
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Database path relative to project root
DB_PATH = Path(__file__).parent.parent.parent / "pubmed-vectors" / "pubmed_data.db"


def normalize_authors():
    """Create a normalized table with one row per author."""
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Create normalized table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles_by_author (
            pmid TEXT,
            title TEXT,
            author TEXT,
            abstract TEXT,
            publication_year INTEGER,
            FOREIGN KEY (pmid) REFERENCES articles(pmid)
        )
    ''')

    # Create index for faster lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_author_lower 
        ON articles_by_author(LOWER(author))
    ''')

    # Check if table already has data
    cursor.execute('SELECT COUNT(*) FROM articles_by_author')
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        logger.info(f"Table articles_by_author already has {existing_count} rows. Skipping normalization.")
        conn.close()
        return

    # Get total count for progress bar
    cursor.execute('SELECT COUNT(*) FROM articles WHERE authors IS NOT NULL AND authors != ""')
    total_articles = cursor.fetchone()[0]

    logger.info(f"Normalizing {total_articles} articles...")

    # Process articles in batches
    batch_size = 10000
    cursor.execute('SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE authors IS NOT NULL AND authors != ""')
    
    batch = []
    processed = 0

    for row in tqdm(cursor.fetchall(), total=total_articles, desc="Normalizing authors"):
        pmid, title, authors_str, abstract, pub_year = row
        
        if not authors_str:
            continue

        # Split authors by comma and clean up
        authors_list = [author.strip() for author in authors_str.split(',') if author.strip()]
        
        # Create one row per author
        for author in authors_list:
            batch.append((pmid, title, author, abstract, pub_year))
            
            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT INTO articles_by_author (pmid, title, author, abstract, publication_year) VALUES (?, ?, ?, ?, ?)',
                    batch
                )
                conn.commit()
                processed += len(batch)
                batch = []

    # Insert remaining batch
    if batch:
        cursor.executemany(
            'INSERT INTO articles_by_author (pmid, title, author, abstract, publication_year) VALUES (?, ?, ?, ?, ?)',
            batch
        )
        conn.commit()
        processed += len(batch)

    logger.info(f"Normalization complete. Created {processed} author-article rows.")
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    normalize_authors()

