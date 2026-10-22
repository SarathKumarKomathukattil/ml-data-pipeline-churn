from pathlib import Path
from ingest import ingest_data
from clean import clean_data
from validate import validate_data
from train import train_model


def get_latest_raw_file():
    raw_dir = Path('data/raw')
    raw_files= list(raw_dir.glob('user_activity_*.csv'))

    if not raw_files:
        raise FileNotFoundError('No raw CSV files found in data/raw')

    latest_file = max(raw_files,key=lambda file:file.name)
    return latest_file



def run_pipeline(file_path):
    raw_df = ingest_data(file_path)
    cleaned_df = clean_data(raw_df)

    is_valid,errors = validate_data(cleaned_df)

    if not is_valid:
        print('\nTraining skipped due to data quality failure.')
        return 

    processed_dir = Path('data/processed')
    processed_dir.mkdir(exist_ok=True)

    raw_file = Path(file_path)
    processed_file = processed_dir/f'{raw_file.stem}_processed.csv'
    cleaned_df.to_csv(processed_file,
                      index=False)

    print(f'\nProcessed data saved to: {processed_file}')
    print("\nData quality passed, ready for model training.")
    train_model(cleaned_df)

if __name__ =="__main__":
    latest_file = get_latest_raw_file()
    run_pipeline(latest_file)