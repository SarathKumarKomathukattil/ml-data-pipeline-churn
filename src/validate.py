import pandas as pd


REQUIRED_COLUMNS = {
    'user_id',
    'snapshot_datetime',
    'age',
    'tenure_months',
    'avg_monthly_spend',
    'avg_session_minutes',
    'support_tickets',
    'days_since_last_login',
    'plan_type',
    'churn'
}

VALID_PLANS = {
    'Basic',
    'Standard',
    'Premium',
}

def validate_data(df:pd.DataFrame):
    '''Validate cleaned customer data before initiating model training'''
    errors = []

    if df.empty:
        errors.append('dataset is empty')
        return False,errors

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f'Missing columns: {missing_columns}')
        return False,errors

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    for column,count in null_counts.items():
        if count>0:
            errors.append(f'{column} contains {count} missing values')

    duplicate_count = df.duplicated(subset=['user_id','snapshot_datetime']).sum()
    if duplicate_count>0:
        errors.append(f"Found {duplicate_count} duplicate userid's")

    if not df['age'].between(18,70).all():
        errors.append('age contains value outside 18-70')

    if not df['tenure_months'].between(1, 72).all():
        errors.append('tenure_months contains value outside 1-72')

    if (df['avg_session_minutes'] < 0).any():
        errors.append('avg_session_minutes contains negative values')

    if (df["avg_monthly_spend"] < 0).any():
        errors.append('avg_monthly_spend contains negative values')

    if (df["support_tickets"] < 0).any():
        errors.append('support_tickets contains negative values')

    if (df["days_since_last_login"] < 0).any():
        errors.append('days_since_last_login contains negative values')

    if not df['plan_type'].isin(VALID_PLANS).all():
        errors.append('plan_type contains invalid options')

    if not df['churn'].isin([0,1]).all():
        errors.append('churn value must be either 0 or 1')

    if errors:
        print(f'\nData validation failed due to errors {errors}')
        return False,errors

    print('\nData validation passed')
    return True,[]









    


