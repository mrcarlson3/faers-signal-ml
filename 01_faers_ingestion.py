import json
import logging
import os
import glob

logging.basicConfig(level=logging.INFO)

def parse_and_flatten(results, drug_pk_counter, reaction_pk_counter, outcome_pk_counter):
    reports, drugs, reactions, outcomes = [], [], [], []

    for row in results:
        # Standardize primary key 
        report_id = row.get('safetyreportid', 'UNKNOWN')
        if report_id == 'UNKNOWN':
            continue
            
        patient = row.get('patient', {})
        age = patient.get('patientonsetage', None)
        try: age = float(age) if age else None
        except ValueError: age = None
            
        sex = patient.get('patientsex', None)
        try: sex = int(sex) if sex else None
        except ValueError: sex = None
            
        reports.append({
            'report_id': report_id,
            'receive_date': row.get('receivedate', None),
            'patient_age': age,
            'patient_sex': sex,
            'serious': row.get('serious', None)
        })
        
        patient_drugs = patient.get('drug', [])
        if isinstance(patient_drugs, dict): patient_drugs = [patient_drugs] # Handle rare case
        for d in patient_drugs:
            char = d.get('drugcharacterization', None)
            try: char = int(char) if char else None
            except ValueError: char = None
                
            drugs.append({
                'drug_id': drug_pk_counter,
                'report_id': report_id,
                'medicinal_product': d.get('medicinalproduct', None),
                'drug_characterization': char
            })
            drug_pk_counter += 1
            
        patient_reactions = patient.get('reaction', [])
        if isinstance(patient_reactions, dict): patient_reactions = [patient_reactions]
        for r in patient_reactions:
            reactions.append({
                'reaction_id': reaction_pk_counter,
                'report_id': report_id,
                'reaction_meddra_pt': r.get('reactionmeddrapt', None)
            })
            reaction_pk_counter += 1
            
        # 4. Extract Outcomes 
        outcome_flags = [
            ('seriousnessdeath', 'DEATH'),
            ('seriousnesshospitalization', 'HOSPITALIZATION'),
            ('seriousnessdisabling', 'DISABLING'),
            ('seriousnesslifethreatening', 'LIFE_THREATENING')
        ]
        
        for flag, code in outcome_flags:
            val = row.get(flag, None)
            if val == '1':
                outcomes.append({
                    'outcome_id': outcome_pk_counter,
                    'report_id': report_id,
                    'outcome_code': code
                })
                outcome_pk_counter += 1

    return reports, drugs, reactions, outcomes, drug_pk_counter, reaction_pk_counter, outcome_pk_counter

def append_to_ndjson(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    RAW_DATA_DIR = './data'
    
    # Reset output files to avoid duplications on re-run
    output_files = ['raw_reports.json', 'raw_drugs.json', 'raw_reactions.json', 'raw_outcomes.json']
    for f in output_files:
        if os.path.exists(f):
            os.remove(f)

    json_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.json'))
    logging.info(f"Found {len(json_files)} extracted JSON partitions in {RAW_DATA_DIR}.")

    drug_pk_counter = 1
    reaction_pk_counter = 1
    outcome_pk_counter = 1
    total_reports = 0

    for idx, filepath in enumerate(json_files):
        try:
            logging.info(f"Processing partition {idx+1}/{len(json_files)}: {os.path.basename(filepath)}")
            with open(filepath, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            
            results = payload.get('results', [])
            if not results:
                continue

            # Process and flatten
            flat_reports, flat_drugs, flat_reactions, flat_outcomes, \
            drug_pk_counter, reaction_pk_counter, outcome_pk_counter = \
                parse_and_flatten(results, drug_pk_counter, reaction_pk_counter, outcome_pk_counter)
            
            total_reports += len(flat_reports)

            # Append to NDJSON iteratively to preserve memory
            append_to_ndjson(flat_reports, 'raw_reports.json')
            append_to_ndjson(flat_drugs, 'raw_drugs.json')
            append_to_ndjson(flat_reactions, 'raw_reactions.json')
            append_to_ndjson(flat_outcomes, 'raw_outcomes.json')
            
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")

    logging.info(f"Successfully processed and flattened {total_reports} reports into NDJSON format.")
