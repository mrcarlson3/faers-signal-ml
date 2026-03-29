# DS 4320 Project 1: FAERS Pharmacovigilance Signal Detection

**Executive Summary**
This project provides a robust, local data ingestion and processing pipeline for the FDA Adverse Event Reporting System (FAERS), enabling scalable machine-learning-driven signal detection. By automating data pulls, organizing the findings into a relational DuckDB database, and exporting engineered features, we aim to rapidly highlight unsafe drug-reaction associations.

**Name:** [Your Name]
**NetID:** [Your NetID]
**DOI:** [Insert DOI Here]

**Links:**
- [Press Release File](press_release.md)
- [Data Directory](data/)
- [Pipeline Source Files](#data-creation)
- [License Information](LICENSE)

## Problem Definition
**Problem Statement:** Identifying previously unknown adverse drug reactions (ADRs) from passive reporting systems like FAERS is painfully slow. Human analysts struggle with vast reporting volumes, frequent duplicate records, and tangled correlations among multiple concomitant drugs.
**Rationale:** Relational restructuring and engineered feature generation using DuckDB and Python allow analysts to harness the entire dataset simultaneously, extracting actionable intelligence rapidly rather than sifting manually.
**Motivation:** Faster and more accurate identification of drug risks ultimately saves lives, avoids severe patient harm, and reduces the tremendous healthcare costs associated with drug-induced injuries.
**Headline:** FDA Safety Signals Unlocked: New ML Pipeline Predicts Adverse Drug Reactions
**Link to Press Release:** [Read our Press Release here](press_release.md)

## Domain Exposition
**Terminology Table:**
| Term | Definition |
|------|------------|
| Pharmacovigilance | The science of detecting, assessing, and preventing adverse effects relating to pharmaceutical products. |
| ADR | Adverse Drug Reaction; any noxious, unintended, and undesired effect that occurs at normal drug doses. |
| FAERS | FDA Adverse Event Reporting System; the primary US repository for post-market drug safety intelligence. |
| Signal Detection | Statistical processing that highlights new or known adverse events potentially caused by a specific medicine. |
| DuckDB | An in-process SQL OLAP database we use to handle the massive FAERS quarterly exports locally. |

**Domain Explanation:**
Pharmacovigilance heavily relies on spontaneous reporting to global databases like FAERS to monitor post-marketing drug safety. The main analytical challenge is separating the true "signals" (causal links between drug and effect) from background "noise" (coincidental events and baseline disease rates). We implement ML approaches that rely on robust feature engineering to unearth these hidden patterns.

**Background Reading Folder:** [background_reading/](background_reading/) *(Placeholder for provided materials)*

**Background Reading Table:**
| Source Title | Relevance | Source URL/Reference |
|--------------|-----------|----------------------|
| FDA FAERS Overview | Understanding the origins, constraints, and structural framework of the data. | [FDA FAERS Page] |
| Introduction to Safety Signal Detection | General statistical background on how true ADR signals are mathematically captured. | [Literature Reference] |

## Data Creation
**Provenance:**
Data originates completely from the FDA through direct bulk downloads and the openFDA interface, released generally as quarterly ASCII datasets.

**Code Provenance Table:**
| Script | Description |
|--------|-------------|
| `01_faers_bulk_download.py` | Retrieves and stages raw FAERS ASCII quarterly dumps from FDA endpoints. |
| `01_faers_ingestion.py` | Parses raw files, standardizes varied formatting, and prepares clean records. |
| `02_duckdb_schema.sql` | Establishes the relational schema and loads our data into `faers_ml.duckdb`. |
| `03_feature_engineering.py` | Pulls formatted tables from DuckDB to create ML-ready features and CSV exports. |

**Bias Identification:**
Spontaneous passive reporting systems universally face immense biases, notably drastic under-reporting, the "Weber effect" (disproportionately high reporting immediately following a drug's launch), and confounding by indication (sicker patients take the drug and thus experience more adverse events). 
**Bias Mitigation:**
Our feature engineering normalizes adverse event counts against a background timeline specific to that drug, mapping proportional reporting ratios (PRR) rather than raw absolute counts. We also stratify models based on reporter types (e.g., healthcare professional vs. consumer) whenever possible.
**Rationale:**
By mathematically adjusting for inherent systemic bias, our downstream ML models process trends denoting true physiological risks rather than simply mapping reporter enthusiasm or market saturation.

## Metadata
**ER Diagram:**
![ER Diagram Placeholder](er_diagram_placeholder.png)

**Data Table List:**
The resulting structural exports that feed into modeling:
| File Name | Description |
|-----------|-------------|
| `faers_reports.csv` | Core administrative and patient demographic details per event report. |
| `faers_drugs.csv` | Drugs listed within the reports, delineated by suspected versus concomitant role. |
| `faers_reactions.csv` | Specific adverse event outcomes coded in MedDRA Preferred Terms. |
| `faers_outcomes.csv` | General consequence of the event (e.g., Hospitalization, Disability, Death). |

**Data Dictionary:**
| Table | Feature / Variable | Description | Data Type | Uncertainty / Missingness Notes |
|-------|--------------------|-------------|-----------|--------------------------------|
| `faers_reports` | `primaryid` | Unique report string matching | VARCHAR | Negligible missingness. |
| `faers_reports` | `age` | Patient age at the time of the event | FLOAT | High missingness; frequently omitted by overwhelmed reporters. |
| `faers_reports` | `sex` | Patient biological sex | VARCHAR | Moderate missingness. |
| `faers_drugs` | `drug_name` | The name or active ingredient | VARCHAR | Text noise exists; spelling variations mandate NLP normalization in the future. |
| `faers_drugs` | `role_cod` | Role in the event (Primary Suspect / Concomitant) | VARCHAR | Usually reliable if originated by healthcare staff. |
| `faers_reactions` | `pt` | Preferred Term detailing symptom | VARCHAR | Highly standardized, but completely dependent on reporter's interpretation. |
| `faers_outcomes` | `outc_cod` | Outcome tag (e.g., DE for Death, HO for Hospital) | VARCHAR | Often blank for non-serious events. |