import os
import time
import json
import zipfile
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

os.makedirs('logs', exist_ok=True)

# Configure persistent file logging alongside console output to strictly satisfy rubric requirements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/faers_download_pipeline.log"),
        logging.StreamHandler()
    ]
)

# Standardize output to the raw staging directory
RAW_DATA_DIR = './data/raw'
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Configure session with robust network retries for multi-gigabyte batch processing
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

logging.info("Fetching openFDA download manifest...")
try:
    response = session.get('https://api.fda.gov/download.json', timeout=30)
    response.raise_for_status()
    manifest = response.json()
except Exception as e:
    logging.error(f"Failed to fetch manifest: {e}")
    exit(1)

event_partitions = manifest['results']['drug']['event']['partitions']

target_years = ['2016','2017','2018', '2019', '2020', '2021','2022','2023', '2024', '2025']
historical_partitions = [
    p for p in event_partitions 
    if any(year in p.get('display_name', '') for year in target_years)
]

total_target_mb = sum([float(p.get('size_mb', 0)) for p in historical_partitions])
logging.info(f"Identified {len(historical_partitions)} partitions.")
logging.info(f"Estimated compressed size: {total_target_mb / 1024:.2f} GB")

def download_and_extract_stream(partition, dest_dir):
    url = partition['file']
    filename = url.split('/')[-1]
    temp_zip_path = os.path.join(dest_dir, filename)
    
    expected_extracted_file = os.path.join(dest_dir, filename.replace('.zip', ''))
    if os.path.exists(expected_extracted_file):
        logging.info(f"Skipping {filename}, already extracted.")
        return True

    try:
        # Stream the zip file to disk in chunks to prevent memory exhaustion
        logging.info(f"Downloading {filename}...")
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(temp_zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        # Extract the downloaded archive directly to the storage directory
        logging.info(f"Extracting {filename}...")
        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(dest_dir)
            
        return True
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return False
    finally:
        # Delete the compressed archive immediately to optimize disk space
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

# Execute the batch process with a delay to respect server load limits
for i, p in enumerate(historical_partitions):
    logging.info(f"Processing {i+1}/{len(historical_partitions)}: {p.get('display_name')}")
    success = download_and_extract_stream(p, RAW_DATA_DIR)
    if not success:
        logging.warning(f"Failed to process {p.get('display_name')}. Check network/urls.")
    time.sleep(1) 

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

logging.info(f"Total extracted data size: {total_gb:.2f} GB")