-- 02_duckdb_schema.sql
-- duckdb SQL DDL to define four linked tables for the FAERS database 
-- and ingest the flattened JSON files from 01_faers_ingestion.py

-- Drop tables if they exist to allow for re-runs
DROP TABLE IF EXISTS faers_outcomes;
DROP TABLE IF EXISTS faers_reactions;
DROP TABLE IF EXISTS faers_drugs;
DROP TABLE IF EXISTS faers_reports;

-- 1. Create Core Reports Table
CREATE TABLE faers_reports (
    report_id VARCHAR PRIMARY KEY,
    receive_date DATE,
    patient_age FLOAT,
    patient_sex INTEGER,
    seriousness_score INTEGER -- This raw "serious" flag will be engineered next
);

-- 2. Create Drugs Table
CREATE TABLE faers_drugs (
    drug_id INTEGER PRIMARY KEY,
    report_id VARCHAR,
    medicinal_product VARCHAR,
    drug_characterization INTEGER,
    FOREIGN KEY (report_id) REFERENCES faers_reports(report_id)
);

-- 3. Create Reactions (Adverse Events) Table
CREATE TABLE faers_reactions (
    reaction_id INTEGER PRIMARY KEY,
    report_id VARCHAR,
    reaction_meddra_pt VARCHAR,
    FOREIGN KEY (report_id) REFERENCES faers_reports(report_id)
);

-- 4. Create Outcomes Table
CREATE TABLE faers_outcomes (
    outcome_id INTEGER PRIMARY KEY,
    report_id VARCHAR,
    outcome_code VARCHAR,
    FOREIGN KEY (report_id) REFERENCES faers_reports(report_id)
);

-- Ingest sample flat JSONs (if already created by the ingestion script)
-- NOTE: In production, the json payload should be parsed correctly using the read_json_auto wrapper

INSERT INTO faers_reports (report_id, receive_date, patient_age, patient_sex, seriousness_score)
SELECT report_id, 
       strptime(receive_date, '%Y%m%d'), -- FAERS date format
       patient_age, 
       patient_sex, 
       TRY_CAST(serious AS INTEGER)
FROM read_json_auto('raw_reports.json');

INSERT INTO faers_drugs
SELECT * FROM read_json_auto('raw_drugs.json');

INSERT INTO faers_reactions
SELECT * FROM read_json_auto('raw_reactions.json');

INSERT INTO faers_outcomes
SELECT * FROM read_json_auto('raw_outcomes.json');

-- Export tables to Parquet files for efficient downstream reading
COPY faers_reports TO 'faers_reports.parquet' (FORMAT PARQUET);
COPY faers_drugs TO 'faers_drugs.parquet' (FORMAT PARQUET);
COPY faers_reactions TO 'faers_reactions.parquet' (FORMAT PARQUET);
COPY faers_outcomes TO 'faers_outcomes.parquet' (FORMAT PARQUET);

