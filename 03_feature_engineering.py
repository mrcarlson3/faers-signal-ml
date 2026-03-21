import duckdb
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def run_feature_engineering(db_connection):
    """
    Connects to the duckdb database, creates the binary Seriousness Score target,
    extracts a joined dataset, and maps basic MedDRA term prevalence.
    """
    logging.info("Starting feature engineering in DuckDB...")

    # Validate that we have data
    count = db_connection.execute("SELECT COUNT(*) FROM faers_reports").fetchone()[0]
    if count == 0:
        logging.error("No reports found in db. Did you run the ingestion and SQL scripts?")
        return

    # Step 1: Engineer Binary Classification Target
    # We update the seriousness_score column using the logic laid out in the requirements:
    # OpenFDA codes 1=Serious, 2=Non-Serious. We map it to Binary: 1=Serious, 0=Non-Serious.
    # Missing values default to 0 (non-serious).
    logging.info("Engineering the 'seriousness_score' binary classification target...")
    db_connection.execute("""
        UPDATE faers_reports
        SET seriousness_score = CASE 
            WHEN seriousness_score = 1 THEN 1
            ELSE 0 
        END
    """)
    
    # Check the newly engineered distribution
    target_dist = db_connection.execute("""
        SELECT seriousness_score, COUNT(*) as count 
        FROM faers_reports 
        GROUP BY seriousness_score
    """).df()
    
    logging.info(f"Target Distribution after engineering:\n{target_dist}")

    # Step 2: Resolve Confounding by indication
    # Filter to primary suspect drugs only (drug_characterization = 1)
    logging.info("Creating materialized modeling view filtered for 'Primary Suspect' drugs...")
    db_connection.execute("""
        CREATE OR REPLACE VIEW v_primary_suspect_reports AS
        SELECT 
            r.report_id,
            r.patient_age,
            r.patient_sex,
            r.seriousness_score,
            d.medicinal_product,
            re.reaction_meddra_pt
        FROM faers_reports r
        JOIN faers_drugs d ON r.report_id = d.report_id
        JOIN faers_reactions re ON r.report_id = re.report_id
        WHERE d.drug_characterization = 1 
          -- Ensure basic data integrity
          AND r.patient_age IS NOT NULL 
    """)

    # Export an example to pandas to demonstrate utility
    df_model_ready = db_connection.execute("SELECT * FROM v_primary_suspect_reports LIMIT 10").df()
    logging.info(f"Sample of Model-Ready Joined Data:\n{df_model_ready.head()}")

if __name__ == "__main__":
    # Create an in-memory DuckDB testing database, load the schema + data, then run transformations
    con = duckdb.connect(database='faers_ml.duckdb', read_only=False)
    
    logging.info("Executing 02_duckdb_schema.sql to build schema and ingest data...")
    with open('02_duckdb_schema.sql', 'r') as f:
        sql_schema = f.read()
    
    # We execute our DDL and insert script dynamically from python to make the demo self-contained
    try:
        con.execute(sql_schema)
        run_feature_engineering(con)
    except Exception as e:
        logging.error(f"Error during execution: {e}")
    finally:
        con.close()
