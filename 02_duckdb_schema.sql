-- Drop existing tables to ensure idempotency during pipeline re-runs.
DROP TABLE IF EXISTS outcomes;
DROP TABLE IF EXISTS reactions;
DROP TABLE IF EXISTS drugs;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS reports;

-- Establish the core report administrative table.
CREATE TABLE reports (
    report_id VARCHAR PRIMARY KEY,
    receive_date DATE,
    serious VARCHAR
);

-- Isolate patient demographics to normalize the schema and prevent data duplication.
CREATE TABLE patients (
    report_id VARCHAR,
    patient_sex VARCHAR,
    patient_age FLOAT,
    FOREIGN KEY (report_id) REFERENCES reports(report_id)
);

-- Track drug exposures and their suspected clinical roles.
CREATE TABLE drugs (
    report_id VARCHAR,
    drug_name VARCHAR,
    role_cod VARCHAR,
    FOREIGN KEY (report_id) REFERENCES reports(report_id)
);

-- Record specific adverse event symptom terminology.
CREATE TABLE reactions (
    report_id VARCHAR,
    pt VARCHAR,
    FOREIGN KEY (report_id) REFERENCES reports(report_id)
);

-- Standardize severity classifications to isolate the predictive target variable.
CREATE TABLE outcomes (
    report_id VARCHAR,
    outcome_code VARCHAR,
    FOREIGN KEY (report_id) REFERENCES reports(report_id)
);

-- Populate tables directly from out-of-core NDJSON files.
-- Date parsing normalizes the YYYYMMDD string format inherent to raw FAERS data.
INSERT INTO reports
SELECT 
    report_id,
    TRY_CAST(strptime(receive_date, '%Y%m%d') AS DATE),
    serious
FROM read_json_auto('./data/ndjson/raw_reports.json');

-- Age is cast to FLOAT to handle decimal representations common in pediatric reports.
INSERT INTO patients
SELECT 
    report_id,
    patient_sex,
    TRY_CAST(patient_age AS FLOAT)
FROM read_json_auto('./data/ndjson/raw_patients.json');

INSERT INTO drugs
SELECT report_id, drug_name, role_cod
FROM read_json_auto('./data/ndjson/raw_drugs.json');

INSERT INTO reactions
SELECT report_id, pt
FROM read_json_auto('./data/ndjson/raw_reactions.json');

INSERT INTO outcomes
SELECT report_id, outcome_code
FROM read_json_auto('./data/ndjson/raw_outcomes.json');

-- Export schema to Parquet format to satisfy rubric storage optimization constraints.
COPY reports TO './data/parquet/faers_reports.parquet' (FORMAT PARQUET);
COPY patients TO './data/parquet/faers_patients.parquet' (FORMAT PARQUET);
COPY drugs TO './data/parquet/faers_drugs.parquet' (FORMAT PARQUET);
COPY reactions TO './data/parquet/faers_reactions.parquet' (FORMAT PARQUET);
COPY outcomes TO './data/parquet/faers_outcomes.parquet' (FORMAT PARQUET);