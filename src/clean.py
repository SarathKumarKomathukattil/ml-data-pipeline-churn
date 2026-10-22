import pandas as pd

def clean_data(raw_df):
    df = raw_df.copy()

    rows_before = len(df)

    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()

    df['user_id'] = df['user_id'].astype('string').str.strip()
    df['plan_type'] = df['plan_type'].astype('string').str.strip().str.title()
    df['snapshot_datetime'] = pd.to_datetime(df['snapshot_datetime'],
                                             errors='coerce',
                                             utc=True)

    rows_removed = rows_before-len(df)
    print(f'Removed {rows_removed} duplicate rows')
    return df




    
