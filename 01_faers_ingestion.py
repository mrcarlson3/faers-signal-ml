import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

def fetch_faers_data(limit=1000):
    """
    Queries the openFDA API and retrieves nested JSON historical reports.
    """
    logging.info(f"Fetching {limit} reports from openFDA...")
    url = f"https://api.fda.gov/drug/event.json?limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    
    return response.json().get('results', [])

def parse_and_flatten(results):
    """
    Parses nested JSON historical reports into flat lists of dictionaries 
    representing the four entities for our DuckDB relational model.
    """
    logging.info("Flattening FAERS JSON payload into relational entities...")
    
    reports = []
    drugs = []
    reactions = []
    outcomes = []
    
    drug_pk_counter = 1
    reaction_pk_counter = 1
    outcome_pk_counter = 1

    for row in results:
        # Standardize primary key 
        report_id = row.get('safetyreportid', 'UNKNOWN')
        if report_id == 'UNKNOWN':
            continue
            
        # 1. Extract Report level data
        patient = row.get('patient', {})
        age = patient.get('patientonsetage', None)
        # Attempt to cast age to float if possible
        try:
            age = float(age) if age else None
        except ValueError:
            age = None
            
        sex = patient.get('patientsex', None)
        try:
            sex = int(sex) if sex else None
        except ValueError:
            sex = None
            
        reports.append({
            'report_id': report_id,
            'receive_date': row.get('receivedate', None),
            'patient_age': age,
            'patient_sex': sex,
            'serious': row.get('serious', None)  # Pre-feature engineering value
        })
        
        # 2. Extract Drug level data
        patient_drugs = patient.get('drug', [])
        for d in patient_drugs:
            char = d.get('drugcharacterization', None)
            try:
                char = int(char) if char else None
            except ValueError:
                char = None
                
            drugs.append({
                'drug_id': drug_pk_counter,
                'report_id': report_id,
                'medicinal_product': d.get('medicinalproduct', None),
                'drug_characterization': char
            })
            drug_pk_counter += 1
            
        # 3. Extract Reaction (MedDRA) level data
        patient_reactions = patient.get('reaction', [])
        for r in patient_reactions:
            reactions.append({
                'reaction_id': reaction_pk_counter,
                'report_id': report_id,
                'reaction_meddra_pt': r.get('reactionmeddrapt', None)
            })
            reaction_pk_counter += 1
            
        # 4. Extract Outcomes (Simulated mapping from top-level seriousness fields)
        # In actual FAERS data, multiple seriousness flags can exist natively.
        outcome_flags = [
            ('seriousnessdeath', 'DEATH'),
            ('seriousnesshospitalization', 'HOSPITALIZATION'),
            ('seriousnessdisabling', 'DISABLING'),
            ('seriousnesslifethreatening', 'LIFE_THREATENING')
        ]
        
        for flag, code in outcome_flags:
            val = row.get(flag, None)
            if val == '1': # 1 indicates the outcome occurred
                outcomes.append({
                    'outcome_id': outcome_pk_counter,
                    'report_id': report_id,
                    'outcome_code': code
                })
                outcome_pk_counter += 1

    return reports, drugs, reactions, outcomes

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    faers_results = fetch_faers_data(limit=100) # Small limit for proof of concept
    flat_reports, flat_drugs, flat_reactions, flat_outcomes = parse_and_flatten(faers_results)
    
    save_to_json(flat_reports, 'raw_reports.json')
    save_to_json(flat_drugs, 'raw_drugs.json')
    save_to_json(flat_reactions, 'raw_reactions.json')
    save_to_json(flat_outcomes, 'raw_outcomes.json')
    
    logging.info(f"Saved {len(flat_reports)} reports to raw JSON output ready for ingestion.")
