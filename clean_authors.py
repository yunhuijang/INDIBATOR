import sqlite3
import re

DB_PATH = "pubmed-vectors/pubmed_data_org.db"
BATCH_SIZE = 50000

# Adapted regex for "Initials Lastname" format
AUTHOR_PATTERN = re.compile(r"^[a-z]{1,4}(?: [a-z]{1,4})* [a-z][a-z'\-]+(?: [a-z'\-]+)*$")

def clean_author(author_raw: str) -> str | None:
    """Clean author to match BigQuery format."""
    # Remove non-letter/space/apostrophe/hyphen, lowercase, collapse whitespace
    cleaned = re.sub(r"[^A-Za-z '\-]", "", author_raw.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()

    # Filter: only keep if matches pattern
    if AUTHOR_PATTERN.match(cleaned):
        return cleaned
    return None

def process_authors(authors_str: str) -> str:
    """Process comma-separated authors, return cleaned comma-separated result."""
    if not authors_str:
        return ''  # Use empty string instead of None to mark as processed
    cleaned = [clean_author(a) for a in authors_str.split(',')]
    cleaned = [c for c in cleaned if c]  # Remove None values
    return ', '.join(cleaned) if cleaned else ''  # Use empty string instead of None

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM articles WHERE author_clean IS NULL")
    total = cursor.fetchone()[0]
    print(f"Total rows to process: {total:,}")

    processed = 0
    while True:
        # Fetch batch
        cursor.execute(
            "SELECT pmid, authors FROM articles WHERE author_clean IS NULL LIMIT ?",
            (BATCH_SIZE,)
        )
        rows = cursor.fetchall()

        if not rows:
            break

        # Process and update
        updates = []
        for pmid, authors in rows:
            author_clean = process_authors(authors)
            updates.append((author_clean, pmid))

        cursor.executemany(
            "UPDATE articles SET author_clean = ? WHERE pmid = ?",
            updates
        )
        conn.commit()

        processed += len(rows)
        print(f"Processed {processed:,} / {total:,} ({100*processed/total:.1f}%)")

    conn.close()
    print("Done!")

if __name__ == "__main__":
    main()
