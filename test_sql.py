import duckdb
import os
import subprocess
from pathlib import Path

# Path to your existing SQLite database
db_path = 'pubmed-vectors/pubmed_data_org.db'
db_path_abs = os.path.abspath(db_path)


# 1. Connect to DuckDB (in-memory)
con = duckdb.connect()

try:
    # 2. Install the SQLite extension so DuckDB can read/write your .db file directly
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute("INSTALL icu;    LOAD icu;")    # For nfc_normalize function

    # 3. Attach the database
    con.execute(f"ATTACH '{db_path_abs}' AS mydb (TYPE SQLITE);")
    
    # 4. Run the Optimized SQL Query
    # DuckDB supports UNNEST, string_split, and regexp_replace natively and rapidly.
    query = """
    CREATE OR REPLACE TABLE mydb.articles_final AS
    WITH authors_expanded AS (
        SELECT
            title, 
            abstract, 
            authors, 
            first_author, 
            last_author,
            -- DuckDB uses string_split and unnest
            unnested_author,
            TRIM(unnested_author) AS author_raw,
            REGEXP_REPLACE(
                LOWER(REGEXP_REPLACE(TRIM(nfc_normalize(unnested_author)), '[^A-Za-z ''\\-]', '', 'g')),
                '\\s+', ' '
            ) AS author_token
        FROM mydb.articles, 
            unnest(string_split(authors, ',')) AS t(unnested_author)
    )
    SELECT 
        title, 
        abstract, 
        authors, 
        first_author, 
        last_author,
        author_token AS author_final,
        TRIM(
            string_split(author_token, ' ')[-1] || ' ' 
        || array_to_string(string_split(author_token, ' ')[:-2], '')
        ) AS authors_clean_2
    FROM authors_expanded
    -- WHERE author_token ~ '^[a-z][a-z''\\-]+(?: [a-z''\\-]+)* [a-z]{1,4}$'
    """
    
    
    print("Executing query...")
    con.execute(query)
    print("Query completed successfully!")
    
finally:
    # Always close the connection
    con.close()
    print("Connection closed.")