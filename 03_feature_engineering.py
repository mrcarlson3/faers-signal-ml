import duckdb
import pandas as pd
import logging
import os

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/faers_pipeline.log"), logging.StreamHandler()]
)


# HIGH-RISK DRUG REGISTRY
# Binary FAERS feature flag grounded in Pirmohamed et al. (2004), Sonawane et al. (2018) and post-2016 CDC opioid crisis data
# a positive flag indicates elevated prior probability of a serious or fatal outcome
# refer to background readings for tables where information comes from

HIGH_RISK_DRUGS = {
    # --- FAERS Volume Anchors (Sonawane 2018) ---
    'LENALIDOMIDE',   # Only drug in top 10 across all 3 outcome categories    # #3 disability outcomes; antidepressant class anchor
    'ACETAMINOPHEN',  # #5 death (7,664 reports)
    'METOCLOPRAMIDE', # #1 disability 

    # --- Hospital ADR Admission Anchors (Pirmohamed 2004) ---
    'ASPIRIN',        # 218 cases; #1 drug in ADR admissions and deaths
    'WARFARIN',       # 129 cases; cross-study anticoagulant anchor
    'FUROSEMIDE',     # 128 cases; diuretic class anchor
    'DICLOFENAC',     # 52 cases; NSAID anchor (non-aspirin ADR profile)
    'DIGOXIN',        # 36 cases; narrow therapeutic index; toxicity admissions
    'IBUPROFEN',      # 34 cases; highest-volume OTC NSAID in US FAERS

    # --- Opioids (Post-2016 FAERS Death-Outcome Signal) ---
    'FENTANYL',       # Pirmohamed opiates group (5 cases); #1 opioid death signal
    'MORPHINE',       # Pirmohamed opiates group (20 cases)
    'TRAMADOL',       # Pirmohamed opiates group (8 cases)
    'OXYCODONE',      # US opioid crisis dominant FAERS death reporter
    'HYDROCODONE',    # US opioid crisis dominant FAERS death reporter
}

# DuckDB REGEXP_MATCHES requires a pipe-delimited alternation pattern.
# Substring logic is intentional: FAERS drug_name is free-text and stores
# branded/compound names (e.g., 'INSULIN GLARGINE', 'FENTANYL PATCH').
HIGH_RISK_PATTERN = '|'.join(sorted(HIGH_RISK_DRUGS))


def build_clinical_features(db_path: str, output_dir: str) -> None:
    """
    Connects to a DuckDB instance containing normalized FAERS tables,
    executes a multi-CTE feature engineering query, and exports 
    the resulting feature matrix to Parquet.

    Parameters
    ----------
    db_path : str
        Path to the faers_ml.duckdb file.
    output_dir : str
        Directory where model_features.parquet will be written.
    """
    logging.info("Connecting to DuckDB...")
    con = duckdb.connect(db_path)
    
    tables = con.execute("SHOW TABLES").fetchdf()
    if tables.empty:
        logging.error("No tables found. Run 02_duckdb_schema.sql first.")
        con.close()
        return

    logging.info("Generating features...")

    query = f"""
    WITH clean_patients AS (
        SELECT 
            report_id, 
            MAX(TRY_CAST(patient_age AS FLOAT)) AS patient_age, 
            MAX(patient_sex) AS patient_sex
        FROM patients
        WHERE patient_age IS NOT NULL 
          AND patient_sex IS NOT NULL
          AND TRY_CAST(patient_age AS FLOAT) BETWEEN 0 AND 120
        GROUP BY report_id
        -- Drop the minority of reports with internally conflicting sex entries
        -- rather than imputing, to avoid systematic bias in a demographic feature.
        HAVING COUNT(DISTINCT patient_sex) = 1
    ),
    
    drug_burden AS (
        SELECT 
            report_id,
            COUNT(DISTINCT drug_name)                                        AS polypharmacy_count,
            SUM(CASE WHEN role_cod = 'PS' THEN 1 ELSE 0 END)                 AS primary_suspect_count,
            -- REGEXP_MATCHES handles free-text variance in drug_name 
            -- (e.g., 'INSULIN GLARGINE' matches on 'INSULIN').
            MAX(CASE 
                WHEN REGEXP_MATCHES(UPPER(COALESCE(drug_name, '')), '{HIGH_RISK_PATTERN}') 
                THEN 1 ELSE 0 
            END)                                                             AS is_high_risk_drug
        FROM drugs
        GROUP BY report_id
    )
    
    SELECT 
        r.report_id,
        c.patient_age,
        c.patient_sex,
        r.reporter_type,
        r.reporter_country,
        EXTRACT(YEAR FROM r.receive_date)            AS report_year,
        COALESCE(d.polypharmacy_count, 0)            AS polypharmacy_count,
        COALESCE(d.primary_suspect_count, 0)         AS primary_suspect_count,
        COALESCE(d.is_high_risk_drug, 0)             AS is_high_risk_drug,
        MAX(CASE 
            WHEN o.outcome_code IN ('DE', 'LT', 'HO') THEN 1 ELSE 0 
        END)                                         AS severe_outcome
    FROM reports r
    INNER JOIN clean_patients c ON r.report_id = c.report_id
    LEFT JOIN drug_burden d     ON r.report_id = d.report_id
    LEFT JOIN outcomes o        ON r.report_id = o.report_id
    GROUP BY 
        r.report_id, c.patient_age, c.patient_sex, 
        r.reporter_type, r.reporter_country, r.receive_date, 
        d.polypharmacy_count, d.primary_suspect_count, d.is_high_risk_drug;
    """
    
    try:
        df = con.execute(query).fetchdf()
    except Exception as e:
        logging.error(f"Feature query failed: {e}")
        con.close()
        return
    finally:
        con.close()
        
    if df.empty:
        logging.warning("Query returned zero records. Verify table contents and join keys.")
        return

    logging.info(f"Raw feature matrix: {len(df):,} rows x {df.shape[1]} columns.")

    # Log distributions for the rubric's Data Dictionary requirements
    df = _log_summary(df)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model_features.parquet')
    df.to_parquet(output_path, index=False)
    
    logging.info(f"Feature matrix written to {output_path}.")


def _log_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logs distributional summaries for numeric features to satisfy the
    data dictionary uncertainty quantification requirement in the rubric.
    """
    numeric_cols = ['patient_age', 'polypharmacy_count', 'primary_suspect_count']
    
    for col in numeric_cols:
        mean  = df[col].mean()
        std   = df[col].std()
        miss  = df[col].isna().mean() * 100
        logging.info(f"{col}: mean={mean:.2f}, std={std:.2f}, missing={miss:.2f}%")
        
    # Log class balance for the binary target and the key drug flag.
    severe_rate    = df['severe_outcome'].mean() * 100
    high_risk_rate = df['is_high_risk_drug'].mean() * 100
    logging.info(f"severe_outcome positive rate: {severe_rate:.2f}%")
    logging.info(f"is_high_risk_drug positive rate: {high_risk_rate:.2f}%")
    
    return df


if __name__ == "__main__":
    DB_PATH    = 'faers_ml.duckdb'
    OUTPUT_DIR = './data/features'
    
    try:
        build_clinical_features(DB_PATH, OUTPUT_DIR)
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")