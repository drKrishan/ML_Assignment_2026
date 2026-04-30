"""
Download Bank Marketing Dataset
"""

import urllib.request
import os

print("Downloading Bank Marketing Dataset...")
print("This may take a minute or two...\n")

# Create data directory
os.makedirs("data", exist_ok=True)

# URL for the dataset
url = "https://raw.githubusercontent.com/pauloemmilio/Bank-Marketing-UCI/master/bank-additional-full.csv"
output_path = "data/bank-additional-full.csv"

try:
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Dataset successfully downloaded to: {output_path}")

    # Verify the download
    import pandas as pd

    df = pd.read_csv(output_path, sep=";")
    print(f"\nDataset Info:")
    print(f"  - Records: {len(df)}")
    print(f"  - Features: {df.shape[1]}")
    print(f"  - Columns: {list(df.columns)[:5]}... (showing first 5)")

except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    print("\nAlternative: Manual Download")
    print("1. Visit: https://archive.ics.uci.edu/ml/datasets/bank+marketing")
    print("2. Download 'bank-additional-full.csv'")
    print(
        "3. Save to: c:\\Users\\Vinoth\\Downloads\\ML_ASSIGNMET\\data\\bank-additional-full.csv"
    )
