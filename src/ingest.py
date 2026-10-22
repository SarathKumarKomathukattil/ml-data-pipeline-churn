import pandas as pd
from pathlib import Path


def ingest_data(path):
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    df = pd.read_csv(file_path)
    print(f'Ingested {len(df)} rows from {file_path.name}')
    return df

