import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
import zipfile
import time

RAW_DATA_DIR = './data'
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)

# Configure session with robust retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Fetch the manifest
print("Fetching openFDA download manifest...")
try:
    response = session.get('https://api.fda.gov/download.json', timeout=30)
    response.raise_for_status()
    manifest = response.json()
except Exception as e:
    print(f"Failed to fetch manifest: {e}")
    exit(1)

event_partitions = manifest['results']['drug']['event']['partitions']

# Task 2.1: Filtering the Manifest for 2024-2025 Data
# Isolate partitions representing only the years 2024 and 2025
target_years = ['2024', '2025']
historical_partitions = []

for partition in event_partitions:
    display_name = partition.get('display_name', '')
    # Check if the partition's display name contains any of the target years
    if any(year in display_name for year in target_years):
        historical_partitions.append(partition)

total_target_mb = sum([float(p.get('size_mb', 0)) for p in historical_partitions])
print(f"Identified {len(historical_partitions)} partitions.")
print(f"Estimated compressed size: {total_target_mb / 1024:.2f} GB")

# Task 2.2: Memory-Efficient Batch Downloading and Extraction
def download_and_extract_stream(partition, dest_dir):
    url = partition['file']
    filename = url.split('/')[-1]
    temp_zip_path = os.path.join(dest_dir, filename)
    
    # Anticipate the extracted filename (usually the zip filename minus .zip)
    expected_extracted_file = os.path.join(dest_dir, filename.replace('.zip', ''))
    if os.path.exists(expected_extracted_file):
        print(f"Skipping {filename}, already extracted.")
        return True

    try:
        # Stream the zip file to disk in chunks to prevent memory exhaustion
        print(f"Downloading {filename}...")
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(temp_zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        # Extract the downloaded archive directly to the storage directory
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(dest_dir)
            
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False
    finally:
        # Delete the compressed archive immediately to optimize disk space
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

# Execute the batch process with a delay to respect server load
for i, p in enumerate(historical_partitions):
    print(f"\nProcessing {i+1}/{len(historical_partitions)}: {p.get('display_name')}")
    success = download_and_extract_stream(p, RAW_DATA_DIR)
    if not success:
        print(f"Failed to process {p.get('display_name')}. Check network/urls.")
    time.sleep(1) 

# Task 2.3: Validating Dataset Scale
def verify_directory_size(directory):
    total_bytes = 0
    # Walk the directory tree to sum the byte size of all extracted JSON files
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes

total_bytes = verify_directory_size(RAW_DATA_DIR)
total_gb = total_bytes / (1024 ** 3)

print(f"Total extracted data size: {total_gb:.2f} GB")
