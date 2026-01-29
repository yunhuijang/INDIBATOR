from google.cloud import bigquery
import config  # This automatically loads .env when imported
from tqdm import tqdm
import duckdb

# Configuration variables
PROJECT_ID = ""
DATASET_NAME = ""


def create_molecules_by_author_table(number):
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    try:
        client.get_dataset(f"{PROJECT_ID}.{DATASET_NAME}")
    except Exception as e:
        print(f"Dataset {PROJECT_ID}.{DATASET_NAME} not found. Creating it now...")
        dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_NAME}")
        dataset.location = "US"  # Must match the location of the tables you are querying
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"✅ Created dataset {PROJECT_ID}.{DATASET_NAME}")


    for number in tqdm([23, 24, 25, 26, 27, 28, 29, 33]):
    # for number in [23]:
        molecules_by_author_query = f'''
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_NAME}.molecules_by_author_{number}` AS
        WITH docs AS (
        SELECT
            d.doc_id,
            d.chembl_id AS document_chembl_id,
            d.doi,
            d.pubmed_id,
            d.title,
            d.authors
        FROM `patents-public-data.ebi_chembl.docs_{number}` AS d
        WHERE d.authors IS NOT NULL
        ),
        -- explode authors; normalize to "lastname initials" token
        authors_expanded AS (
        SELECT
            doc_id, document_chembl_id, doi, pubmed_id, title,
            TRIM(a) AS author_raw,
            -- keep letters, spaces, apostrophes, hyphens; collapse whitespace; lowercase
            REGEXP_REPLACE(
            LOWER(REGEXP_REPLACE(TRIM(a), r'[^A-Za-z \\'\\-]', '')),
            r'\\s+', ' '
            ) AS author_token
        FROM docs, UNNEST(SPLIT(authors, ',')) AS a
        ),
        -- keep tokens that look like "lastname initials" (allow multi-word last names)
        authors_filtered AS (
        SELECT *
        FROM authors_expanded
        WHERE REGEXP_CONTAINS(author_token, r'^[a-z][a-z\\'\\-]+(?: [a-z\\'\\-]+)* [a-z]{{1,4}}$')
        ),
        -- map docs → molregno
        acts AS (
        SELECT a.doc_id, COALESCE(a.molregno, cr.molregno) AS molregno
        FROM `patents-public-data.ebi_chembl.activities_{number}` AS a
        LEFT JOIN `patents-public-data.ebi_chembl.compound_records_{number}` AS cr
            ON cr.record_id = a.record_id
        WHERE COALESCE(a.molregno, cr.molregno) IS NOT NULL
        ),
        mol AS (
        SELECT
            af.author_token AS author,
            af.author_raw,
            af.doc_id,
            af.document_chembl_id,
            af.doi,
            af.pubmed_id,
            af.title,
            cs.standard_inchi_key AS inchi_key,
            cs.canonical_smiles   AS smiles
        FROM authors_filtered AS af
        JOIN acts ac
            ON ac.doc_id = af.doc_id
        JOIN `patents-public-data.ebi_chembl.compound_structures_{number}` AS cs
            ON cs.molregno = ac.molregno
        WHERE cs.canonical_smiles IS NOT NULL
            AND cs.standard_inchi_key IS NOT NULL
        )
        -- dedupe per (author, inchi)
        SELECT author, author_raw, inchi_key, smiles,
            document_chembl_id, doi, pubmed_id, title
        FROM (
        SELECT m.*,
                ROW_NUMBER() OVER (PARTITION BY author, inchi_key ORDER BY document_chembl_id) AS rn
        FROM mol m
        )
        WHERE rn = 1;
        '''
        print("Running query to create table...")
        
        
        # job = client.query(molecules_by_author_query)
        # try:
        #     job.result()  # Wait for completion
        #     print("✅ Query completed!")
        #     if job.errors:
        #         print(f"⚠️  Job had errors: {job.errors}")
        # except Exception as e:
        #     print(f"❌ Query failed: {e}")
        #     if job.errors:
        #         print(f"   Job errors: {job.errors}")
        #     raise
        # result = job.result()
        # df = result.to_dataframe()
        # df.to_parquet(f"data/molecules_by_author_{number}.parquet", index=False)
        # Query the created table to get row count
        count_query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.molecules_by_author_{number}`"
        count_job = client.query(count_query)
        count_result = count_job.result()
        df = count_result.to_dataframe()
        df.to_parquet(f"data/molecules_by_author_{number}.parquet", index=False)
        print(f"📊 {number} Table has {len(df):,} rows")
        
def clean_authors_abstract_table():
    """
    Create a table which 1) expands authors into single author rows, 2) normalizes author names to "lastname firstname_initials" format
    """
    # Path to your existing SQLite database
    db_path = 'pubmed-vectors/pubmed_data_org.db'

    # 1. Connect to DuckDB (in-memory)
    con = duckdb.connect()

    # 2. Install the SQLite extension so DuckDB can read/write your .db file directly
    con.execute("INSTALL sqlite; LOAD sqlite;")

    # 3. Attach your SQLite database to DuckDB
    con.execute(f"ATTACH '{db_path}' AS mydb (TYPE SQLITE);")

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
                LOWER(REGEXP_REPLACE(TRIM(unnested_author), '[^A-Za-z ''\\-]', '')),
                '\\s+', ' ', 'g'
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
        author_raw,
        author_token AS author_final,
        TRIM(
            string_split(author_token, ' ')[-1] || ' ' 
        || array_to_string(string_split(author_token, ' ')[:-2], '')
        ) AS authors_clean_2
    FROM authors_expanded
    -- WHERE author_token ~ '^[a-z][a-z''\\-]+(?: [a-z''\\-]+)* [a-z]{1,4}$'
    """
    print("Running optimized SQL query via DuckDB...")
    con.execute(query)
    print("Done! Table created in your SQLite database.")

# create_molecules_by_author_table()
clean_authors_abstract_table()