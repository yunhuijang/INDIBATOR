import sqlite3
import pandas as pd
import os
from src.evaluate.lead.utils_lead import compute_docking_scores
DB_PATH = "pubmed-vectors/pubmed_data_org.db"

def update_database_with_authors(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Identify Table Name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
    table_name = cursor.fetchone()[0]
    print(f"Targeting table: '{table_name}'")

    # 2. Add New Columns (Safe 'try-except' in case they already exist)
    print("Adding new columns...")
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN first_author TEXT")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN last_author TEXT")
    except sqlite3.OperationalError:
        print("Columns already exist. Proceeding to update values...")

    # 3. Fetch Data (rowid is the fastest way to reference rows for updates)
    print("Reading existing data...")
    df = pd.read_sql(f"SELECT rowid, authors FROM {table_name}", conn)

    if df.empty:
        print("Table is empty. Nothing to update.")
        conn.close()
        return

    # 4. Calculate First and Last authors
    # Split by comma
    authors_split = df['authors'].str.split(',')
    
    # Extract
    df['first_author'] = authors_split.str[0].str.strip()
    df['last_author'] = authors_split.str[-1].str.strip()

    # Drop rows where authors might be missing (optional, prevents errors)
    update_data = df.dropna(subset=['first_author', 'last_author'])

    # Convert to list of tuples for SQL: [(first, last, rowid), ...]
    # We must ensure the order matches the SQL placeholder order below
    data_to_update = list(zip(
        update_data['first_author'], 
        update_data['last_author'], 
        update_data['rowid']
    ))

    # 5. Bulk Update
    print(f"Updating {len(data_to_update)} rows...")
    
    update_sql = f"""
        UPDATE {table_name} 
        SET first_author = ?, 
            last_author = ? 
        WHERE rowid = ?
    """
    
    # executemany is significantly faster than a loop
    cursor.executemany(update_sql, data_to_update)
    
    conn.commit()
    print("Success! Database updated.")
    
    # Verify
    print("\n--- Verification (First 5 Rows) ---")
    cursor.execute(f"SELECT authors, first_author, last_author FROM {table_name} LIMIT 5")
    for row in cursor.fetchall():
        print(row)

    conn.close()


def test_docking():
    smiles_list= ['COc1[nH]c2cccc3c2c1C(C)N(C)C3=O', 'COc1[nH]c2cccc3c2c1CN(C(C)C)C3=O', 'COc1[nH]c2cccc3c2c1CN(CCN)C3=O', 'COc1[nH]c2cccc3c2c1C(=O)CNC3=O', 'COc1[nH]c2cccc3c2c1CCN(C(C)=O)C3=O']
    docking_scores = compute_docking_scores('parp1', smiles_list)
    print(docking_scores)

# --- Run the update ---
# update_database_with_authors(DB_PATH)
test_docking()