# DS 4320 Project 1: FAERS Pharmacovigilance Signal Detection

**Executive Summary**
This project provides a robust, local data ingestion and processing pipeline for the FDA Adverse Event Reporting System (FAERS), enabling scalable machine-learning-driven signal detection. By automating bulk data pulls, organizing the findings into a relational DuckDB database, and exporting engineered clinical features, we predict severe patient outcomes based on demographic and polypharmacy risk factors.

| Project Identity | Resource Links |
| :--- | :--- |
| **Name:** Michael Carlson | [Press Release File](press_release.md) |
| **NetID:** mjy7nw | [UVA OneDrive Data Directory](https://myuva-my.sharepoint.com/:f:/g/personal/mjy7nw_virginia_edu/IgBN5u2lUrCQQp4yvMHYp_ykAWy9Ktwu-TP16ULtfDB8S9g?e=oAYx0b) |
| **DOI:** [Insert DOI Badge Here] | [Pipeline Notebook](pipeline.ipynb) |
| **License:** [MIT](LICENSE) | [Pipeline Markdown](pipeline.md) |


## Problem Definition
**General Problem:** 8. Clinical drug trials
**Specific Problem Statement:** Can we predict life-threatening adverse drug events across the entire FAERS population by combining a patient's demographic vulnerability, polypharmacy burden, and exposure to a deterministic registry of historically high-risk drug classes (e.g., NSAIDs, DOACs, Opioids)?
**Rationale:** Pivoting from a narrow medical hypothesis (e.g., "Is Ibuprofen dangerous?") to a generalized Universal Triage Engine allows the model to learn the underlying systemic risk factors of a patient. Including drug identity as a binary clinical prior (`is_high_risk_drug`) merges mechanistic pharmacological ground-truth with population-level generalizability.
**Motivation:** The FDA receives millions of post-market adverse event reports, creating an impossible manual review backlog. A high-precision triage engine that accurately flags severe hospitalizations allows regulators to prioritize life-threatening cases, overcoming the "alert fatigue" inherent in massive passive reporting systems.
**Headline:** AI Universal Triage Engine Predicts Life-Threatening Adverse Drug Events
**Link to Press Release:** [Read Press Release here](press_release.md)

## Domain Exposition
**Terminology Table:**
| Term | Definition |
|------|------------|
| ADR / ADE | Adverse Drug Reaction (harmful, unintended response to a normal dose) / Adverse Drug Event (broader; includes injuries from medication errors). |
| FAERS | FDA Adverse Event Reporting System; the primary US repository for post-market drug safety intelligence. |
| Pharmacovigilance | The science of detecting, assessing, understanding, and preventing adverse effects of medicines. |
| Signal Detection | Identifying a statistical association between a drug and an adverse event warranting further investigation. |
| PRR / ROR / EBGM | Statistical disproportionality metrics used to compare the proportion or odds of specific ADRs against background rates. |
| MedDRA | Medical Dictionary for Regulatory Activities; standardized terminology classifying adverse events. |
| Primary Suspect Drug | The specific drug the reporter believes caused the adverse event (`drugcharacterization = 1`). |
| Serious Outcome | Death, hospitalization, life-threatening condition, disability, or congenital anomaly. |
| Polypharmacy | The concurrent use of multiple medications by a patient, which exponentially increases the risk of ADRs. |
| DuckDB | An in-process SQL OLAP database used to handle massive FAERS historical datasets out-of-core. |

**Domain Explanation:**
This project sits at the intersection of public health informatics, regulatory science, and machine learning. Within pharmacovigilance, monitoring post-market drug safety via voluntary FDA reports relies on statistical disproportionality to flag abnormally frequent drug-event pairs at the population level. This project reframes that workflow. Instead of asking if a reaction is historically overrepresented, a predictive model calculates the immediate, individualized risk of a life-threatening outcome based on specific patient, drug, and reaction features.

**Background Reading Folder:** [OneDrive Literature Repository](https://myuva-my.sharepoint.com/:f:/r/personal/mjy7nw_virginia_edu/Documents/3Y/3YS/DS%204320/proj1/background%20readings?csf=1&web=1&e=AA0LG1) 

**Background Reading Table:**
| Source Title | Relevance | Source URL/Reference |
|--------------|-----------|----------------------|
| Serious Adverse Drug Events Reported to the FDA: Analysis of the FAERS 2006-2014 Database (Sonawane et al., 2018) | Documents a 2-fold increase in serious ADEs (including 244,000+ deaths), highlighting the urgent need for scalable safety triage. | [PDF Link](https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQDe0jmOSLzhSr1eDOacWqRhAbEqiufxMJ-jSZqnh560cEE?e=Jzkpft) |
| Integration of FAERS, DrugBank and SIDER Data for Machine Learning-based Detection of ADRs (Schreier et al., 2024) | Demonstrates that ML models combining FAERS data with engineered features achieve higher recall/precision than conventional methods. | [PDF Link](https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQD8MzJPZoIkQ63AL3kZ4WFWAdCfCPBgaHhy-9j3BX9jKbA?e=cMzhcd) |
| Developing an AI-Guided Signal Detection in the FAERS (Al-Azzawi et al., 2023) | Proof-of-concept showing ML frameworks can automate control group selection and dismiss false-positive signals systematically. | [PDF Link](https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQDGL7CkIzbsToAf4tsKPQ3LAQRbc_ZO6M2ngrfHGcKLC3c?e=7hZxYi) |
| A Pilot, Predictive Surveillance Model in Pharmacovigilance Using ML Approaches (Ferreira et al., 2024) | Details an ML gradient boosting approach that demonstrated acceptable accuracy and detected a true safety signal six months earlier than human review. | [PDF Link](https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQAVExTS_gzqQ6yJlajvAHmHAeAsYr9tiwyImHFiJmPLJCs?e=aoBugd) |
| Adverse drug reactions as cause of admission to hospital: prospective analysis (Pirmohamed et al., 2004) | Foundational study establishing that 6.5% of hospital admissions are caused by ADRs, carrying a massive projected financial and mortality burden. | [PDF Link](https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQBaLTOFWaU-TZLyFK8bv4qDAaOFRHf4ST50G3F-TGB0-tI?e=R56mwG) |

## Data Creation

> ### 🛠 Data Provenance & Reproducibility
> All data is pulled directly from the **OpenFDA API** using `01_faers_bulk_download.py`. The pipeline is designed to be **reproducible**; running the ingestion scripts will refresh the local `faers_ml.duckdb` without duplicating records. All engineered artifacts are stored in the [UVA OneDrive Directory](https://myuva-my.sharepoint.com/:f:/g/personal/mjy7nw_virginia_edu/IgDpgQQ1-uZ8TbiHbMXPD7xaAX6MqHnvhya5nWgsXy54iFU?e=Uaq5f2) to ensure persistent access and auditability.

**Code Provenance Table:**
| Script | Description | Link |
|--------|-------------|------|
| `01_faers_bulk_download.py` | Retrieves and stages raw FAERS JSON archives. | [GitHub](https://github.com/mrcarlson3/faers-signal-ml/blob/main/01_faers_bulk_download.py) |
| `01_faers_ingestion.py` | Flattens nested JSON hierarchies into relational NDJSON files. | [GitHub](https://github.com/mrcarlson3/faers-signal-ml/blob/main/01_faers_ingestion.py) |
| `02_duckdb_schema.sql` | Establishes the relational schema and ingests data into `faers_ml.duckdb`. | [GitHub](https://github.com/mrcarlson3/faers-signal-ml/blob/main/02_duckdb_schema.sql) |
| `03_feature_engineering.py` | Executes CTEs to build clinical proxy features and exports to Parquet. | [GitHub](https://github.com/mrcarlson3/faers-signal-ml/blob/main/03_feature_engineering.py) |

**Bias Identification:**
Spontaneous passive reporting systems face immense biases, notably drastic under-reporting and "confounding by indication" (sicker patients take more drugs and experience more adverse events). Missing data is not missing at random; severe outcomes are more likely to have complete reporting data than minor events.

> ### ⚖️ Analytic Rigor: Bias Mitigation
> To address the **masking effects** inherent in FAERS, we utilize `balanced_subsample` class weighting, which adjusts weights at the bootstrap level to ensure rare severe signals are not drowned out by non-serious reports. Furthermore, we use **Stratified Sampling** to ensure the rare outcome distribution is preserved across training and testing splits.

## Metadata
**ER Diagram:**
![ER Diagram](./images/ER_Diagram.png)


**Data Table List:**
The resulting structural tables inside DuckDB, exported as Parquet format:
| File Name | Description |
|-----------|-------------|
| `faers_reports.parquet` | Core administrative details and receive dates per event report. |
| `faers_patients.parquet` | Normalized patient demographic profiles (age and sex). |
| `faers_drugs.parquet` | Drugs listed within the reports, delineated by suspected versus concomitant role. |
| `faers_reactions.parquet` | Specific adverse event outcomes coded in MedDRA Preferred Terms. |
| `faers_outcomes.parquet` | Categorical indicators of severe event consequences (e.g., DEATH, HOSPITALIZATION). |

**Data Dictionary:**
| Table | Feature | Description | Data Type | Example | Uncertainty / Missingness Notes |
|-------|---------|-------------|-----------|---------|--------------------------------|
| `reports` | `report_id` | Unique report string matching | VARCHAR | "1000452-1" | Negligible missingness. |
| `reports` | `receive_date` | Date the report was filed | DATE | 2024-01-15 | High reliability. |
| `patients` | `patient_age` | Patient age at the time of the event | FLOAT | 45.5 | High missingness; standard deviation of ~20 years. |
| `patients` | `patient_sex` | Patient biological sex | VARCHAR | "1" | Moderate missingness. |
| `drugs` | `drug_name` | The name or active ingredient | VARCHAR | "ASPIRIN" | Text noise exists; spelling variations present. |
| `drugs` | `role_cod` | Role in the event (Primary Suspect / Concomitant) | VARCHAR | "PS" | Usually reliable if originated by healthcare staff. |
| `reactions` | `pt` | Preferred Term detailing symptom | VARCHAR | "Nausea" | Highly standardized terminology, but dependent on reporter judgment. |
| `outcomes` | `outcome_code` | Categorical outcome tag | VARCHAR | "DEATH" | Blank for non-serious events; biased toward complete reporting for fatal events. |
