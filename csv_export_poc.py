import requests
import pandas as pd
import uuid

# 1. Fetch a batch of records from openFDA
url = "https://api.fda.gov/drug/event.json?limit=100"
response = requests.get(url)
data = response.json().get('results', [])

# 2. Initialize lists to hold our relational rows
reports, drugs, reactions, outcomes = [], [], [], []

drug_pk_counter = 1
reaction_pk_counter = 1
outcome_pk_counter = 1

# 3. Parse the nested JSON into distinct entities
for item in data:
    # Use the FDA's safetyreportid as the primary key, fallback to a UUID
    report_id = item.get('safetyreportid', str(uuid.uuid4()))
    patient = item.get('patient', {})
    
    # --- Table 1: Reports ---
    # Map seriousness to score (1=serious, 0=non-serious), defaulting to 0
    serious_raw = item.get('serious', '2')
    seriousness_score = 1 if serious_raw == '1' else 0

    # Extract age and sex safely
    try:
        patient_age = float(patient.get('patientonsetage')) if patient.get('patientonsetage') else None
    except ValueError:
        patient_age = None
        
    try:
        patient_sex = int(patient.get('patientsex')) if patient.get('patientsex') else None
    except ValueError:
        patient_sex = None

    reports.append({
        'report_id': report_id,
        'receive_date': item.get('receivedate', None),
        'patient_age': patient_age,
        'patient_sex': patient_sex,
        'seriousness_score': seriousness_score
    })
    
    # --- Table 2: Drugs ---
    for d in patient.get('drug', []):
        try:
            drug_char = int(d.get('drugcharacterization')) if d.get('drugcharacterization') else None
        except ValueError:
            drug_char = None
            
        drugs.append({
            'drug_id': drug_pk_counter,
            'report_id': report_id,
            'medicinal_product': d.get('medicinalproduct', None),
            'drug_characterization': drug_char
        })
        drug_pk_counter += 1
        
    # --- Table 3: Reactions ---
    for r in patient.get('reaction', []):
        reactions.append({
            'reaction_id': reaction_pk_counter,
            'report_id': report_id,
            'reaction_meddra_pt': r.get('reactionmeddrapt', None)
        })
        reaction_pk_counter += 1
        
    # --- Table 4: Outcomes ---
    # Outcomes are often boolean flags at the root level in openFDA
    # We map them similarly to 01_faers_ingestion.py
    outcome_flags = [
        ('seriousnessdeath', 'DEATH'),
        ('seriousnesshospitalization', 'HOSPITALIZATION'),
        ('seriousnessdisabling', 'DISABLING'),
        ('seriousnesslifethreatening', 'LIFE_THREATENING')
    ]
    for flag, code in outcome_flags:
        if item.get(flag) == '1':
            outcomes.append({
                'outcome_id': outcome_pk_counter,
                'report_id': report_id,
                'outcome_code': code
            })
            outcome_pk_counter += 1

# 4. Convert lists to DataFrames and export to CSV
pd.DataFrame(reports).to_csv('faers_reports.csv', index=False)
pd.DataFrame(drugs).to_csv('faers_drugs.csv', index=False)
pd.DataFrame(reactions).to_csv('faers_reactions.csv', index=False)
pd.DataFrame(outcomes).to_csv('faers_outcomes.csv', index=False)

print("ETL Complete: Generated 4 relational CSV files matching DuckDB schemas (Reports, Drugs, Reactions, Outcomes).")
