import duckdb
import pandas as pd
import logging
import os

# Configure logging to maintain pipeline traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("faers_pipeline.log"), logging.StreamHandler()]
)

def build_clinical_features(db_path, output_dir):
    logging.info("Connecting to DuckDB to engineer clinical features...")
    con = duckdb.connect(db_path)
    
    # Verify database state before executing complex joins
    tables = con.execute("SHOW TABLES").fetchdf()
    if tables.empty:
        logging.error("No tables found in database. Run ingestion and schema scripts first.")
        return

    logging.info("Executing CTE feature generation for High-Risk Cohort (Warfarin/NSAIDs)...")
    
    # This query implements Pivot 1: Restricting the cohort to patients where 
    # Warfarin or Ibuprofen is the primary suspect, then calculating their polypharmacy burden.
    # TRY_CAST safely handles VARCHAR age representations preventing Binder exceptions.
    query = """
    WITH target_reports AS (
        -- Isolate cohort to patients where Warfarin or NSAIDs are the primary suspect
        SELECT DISTINCT report_id
        FROM drugs
        WHERE (UPPER(drug_name) LIKE '%WARFARIN%' OR UPPER(drug_name) LIKE '%IBUPROFEN%')
          AND role_cod = 'PS'
    ),
    patient_cohort AS (
        SELECT 
            r.report_id,
            MAX(TRY_CAST(p.patient_age AS FLOAT)) AS patient_age,
            MAX(p.patient_sex) AS patient_sex,
            MAX(CASE WHEN o.outcome_code IN ('DEATH', 'LIFE_THREATENING', 'HOSPITALIZATION') THEN 1 ELSE 0 END) AS severe_outcome
        FROM target_reports t
        JOIN reports r ON t.report_id = r.report_id
        INNER JOIN patients p ON r.report_id = p.report_id
        LEFT JOIN outcomes o ON r.report_id = o.report_id
        WHERE p.patient_age IS NOT NULL 
          AND p.patient_sex IS NOT NULL
          AND TRY_CAST(p.patient_age AS FLOAT) BETWEEN 0 AND 120
        GROUP BY r.report_id
    ),
    drug_burden AS (
        -- Calculate the full polypharmacy burden (all concurrent drugs) for this specific cohort
        SELECT 
            d.report_id,
            COUNT(DISTINCT d.drug_name) AS polypharmacy_count
        FROM drugs d
        JOIN target_reports t ON d.report_id = t.report_id
        GROUP BY d.report_id
    )
    SELECT 
        c.report_id,
        c.patient_age,
        c.patient_sex,
        COALESCE(d.polypharmacy_count, 0) AS polypharmacy_count,
        c.severe_outcome
    FROM patient_cohort c
    LEFT JOIN drug_burden d ON c.report_id = d.report_id;
    """
    
    df_features = con.execute(query).fetchdf()
    
    logging.info(f"Extracted {len(df_features)} viable patient records for the high-risk cohort.")
    
    if df_features.empty:
        logging.warning("No records matched the cohort criteria. Verify your raw JSON date ranges contain these drugs.")
        return

    # Binarize demographic variables to ensure compatibility with scikit-learn estimators
    df_features['patient_sex_binary'] = df_features['patient_sex'].map({'1': 1, 'M': 1, '2': 0, 'F': 0}).fillna(-1)
    df_features.drop(columns=['patient_sex'], inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model_features.parquet')
    
    # Export the finalized dataset to Parquet to optimize I/O for the downstream ML notebook
    df_features.to_parquet(output_path, index=False)
    logging.info(f"Engineered features successfully written to {output_path}")
    
    con.close()

if __name__ == "__main__":
    DB_PATH = 'faers_ml.duckdb'
    OUTPUT_DIR = './data/features'
    
    try:
        build_clinical_features(DB_PATH, OUTPUT_DIR)
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")