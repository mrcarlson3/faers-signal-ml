import requests
import pandas as pd

# 1. Fetch proof-of-concept data from openFDA
url = "https://api.fda.gov/drug/event.json?limit=100"
response = requests.get(url)
data = response.json()['results']

# 2. Normalize the nested patient data into a flat DataFrame
df_reports = pd.json_normalize(data)

# Confirm successful ingestion
print(f"Successfully loaded {len(df_reports)} FAERS reports.")

# 1. Isolate and clean the target variable
if 'serious' in df_reports.columns:
    df_reports['seriousness_score'] = df_reports['serious'].fillna('2').astype(int)
    
    # 2. Map openFDA coding (1=Serious, 2=Non-Serious) to binary (1=Serious, 0=Non-Serious)
    df_reports['seriousness_score'] = df_reports['seriousness_score'].apply(lambda x: 1 if x == 1 else 0)

    # 3. Output the summary statistics to prove viability
    summary = df_reports['seriousness_score'].value_counts()
    print("Proof of Concept - Seriousness Score Distribution:")
    print(f"Serious (1): {summary.get(1, 0)}")
    print(f"Non-Serious (0): {summary.get(0, 0)}")
else:
    print("Error: 'serious' column not found in the API response.")
