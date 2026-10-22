import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime,timezone


rng = np.random.default_rng(42)
NUM_ROWS = 10000
snapshot_time = datetime.now(timezone.utc)

df = pd.DataFrame({
    'user_id': [f'USER_{i:05d}' for i in range(1,NUM_ROWS+1)],
    'snapshot_datetime': snapshot_time,
    'age': rng.integers(18,71,NUM_ROWS),
    'tenure_months': rng.integers(1,73,NUM_ROWS),
    'avg_monthly_spend': np.round((rng.uniform(10,200,NUM_ROWS)),2),
    'avg_session_minutes': rng.integers(2,90,NUM_ROWS),
    'support_tickets': rng.poisson(1.5,NUM_ROWS),
    'days_since_last_login': rng.integers(0,90,NUM_ROWS),
    'plan_type': rng.choice(['Basic','Standard','Premium'],NUM_ROWS,p=[0.4,0.4,0.2])

})

plan_effect = df['plan_type'].map({
    'Basic': 0.25,
    'Standard': 0.0,
    'Premium': -0.25
})

churn_score = (
    -2.0
    + 0.04 * df['days_since_last_login']
    + 0.25 * df['support_tickets']
    - 0.02 * df['tenure_months']
    -0.015 * df['avg_session_minutes']
    - 0.003 * df['avg_monthly_spend']
    + plan_effect
)

churn_probability = 1/(1 + np.exp(-churn_score))

df['churn'] = rng.binomial(1,churn_probability)

print(df.head(5))

print("\nChurn counts:")
print(df['churn'].value_counts())

print("\nChurn percentage:")
print(df['churn'].value_counts(normalize=True) * 100)

Path('data/raw').mkdir(exist_ok=True)

file_name = f'data/raw/user_activity_{snapshot_time.strftime('%Y-%m-%d')}.csv'
df.to_csv(file_name,
          index=False)

print(f'\nRaw data saved to {file_name}')