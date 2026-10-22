# ML Data Pipeline for Churn Prediction

This project implements a small end-to-end machine learning data pipeline for a churn prediction use case.

The scenario is that a new CSV containing user activity data arrives each day. The pipeline ingests the latest file, cleans and validates the data, stores the processed dataset, and only starts model training if the data-quality checks pass.

The focus of the project is the **data pipeline and validation gate**, rather than model optimization.

## Pipeline Flow

```text
Generate daily CSV
        |
        v
data/raw/
        |
        v
Ingest
        |
        v
Clean
        |
        v
Validate
   |          |
 FAIL        PASS
   |          |
   v          v
 STOP     Save processed CSV
              |
              v
        Train Logistic Regression
              |
              v
        Save trained model
```

The main orchestration is handled by `pipeline.py`.

If validation fails, the pipeline stops before model training.

## Project Structure

```text
ml-data-pipeline-churn/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── generate_data.py
│   ├── ingest.py
│   ├── clean.py
│   ├── validate.py
│   ├── train.py
│   └── pipeline.py
│
├── tests/
│   ├── test_validate.py
│   └── test_pipeline.py
│
├── pytest.ini
├── requirements.txt
└── README.md
```

## Dataset

For this exercise, I generate a synthetic daily user-activity dataset with 10,000 rows.

Each generated file is date-stamped:

```text
data/raw/user_activity_YYYY-MM-DD.csv
```

The dataset contains:

| Column | Description |
| --- | --- |
| `user_id` | Unique user identifier |
| `snapshot_datetime` | Time the daily snapshot was generated |
| `age` | User age |
| `tenure_months` | Number of months the user has been active |
| `avg_monthly_spend` | Average monthly spend |
| `avg_session_minutes` | Average session duration |
| `support_tickets` | Number of support tickets |
| `days_since_last_login` | Days since the user's last login |
| `plan_type` | Basic, Standard, or Premium |
| `churn` | Binary target: 0 = active, 1 = churned |

The churn label is generated from a probability based on user behavior such as inactivity, support tickets, tenure, session duration, spending, and plan type.

## Data Generation

Run:

```bash
python src/generate_data.py
```

This creates a new raw CSV in:

```text
data/raw/
```

The filename automatically uses the current date.

## Ingestion

`ingest.py` reads the CSV provided by the main pipeline and checks that the file exists before loading it with pandas.

The ingestion module does not perform cleaning or model training.

## Cleaning

`clean.py` performs a small set of cleaning steps before validation:

- removes duplicate rows
- strips whitespace from column names
- cleans `user_id`
- normalizes `plan_type`
- converts `snapshot_datetime` to a UTC datetime

Invalid datetime values are converted to missing values so they can be caught during validation.

## Data Validation

`validate.py` acts as the data-quality gate.

The following checks are performed:

- dataset is not empty
- all required columns exist
- required fields do not contain missing values
- no duplicate `user_id` + `snapshot_datetime` combinations
- age is between 18 and 70
- tenure is between 1 and 72 months
- session duration is not negative
- monthly spend is not negative
- support-ticket count is not negative
- days since last login is not negative
- plan type is one of `Basic`, `Standard`, or `Premium`
- churn is either `0` or `1`

If any check fails, validation returns `False` and the pipeline stops before training.

## Daily File Selection

The pipeline looks inside:

```text
data/raw/
```

for files matching:

```text
user_activity_*.csv
```

Because the filenames follow the `YYYY-MM-DD` format, the latest dated file can be selected automatically.

This means the code does not need to be changed every day when a new CSV arrives.

## Model Training

If validation passes, the cleaned dataset is saved to:

```text
data/processed/
```

with a filename such as:

```text
user_activity_2026-10-23_processed.csv
```

The training step uses a scikit-learn pipeline containing:

- `StandardScaler` for numerical features
- `OneHotEncoder` for `plan_type`
- `LogisticRegression` as the classifier

The data is split into training and test sets using an 80/20 split with stratification on the churn target.

The trained preprocessing and classification pipeline is saved as:

```text
models/churn_pipeline.joblib
```

Logistic Regression was intentionally kept as a simple baseline because the main goal of this exercise is the data pipeline and conditional training workflow.

## Example Model Result

One pipeline run produced:

```text
Accuracy:  0.789
Recall:    0.303
Precision: 0.599
F1 Score:  0.402
ROC-AUC:   0.782
```

I kept the default classification threshold because model tuning was outside the main scope of this exercise.

## Running the Full Pipeline

From the project root:

```bash
python src/generate_data.py
python src/pipeline.py
```

`pipeline.py` automatically selects the latest raw CSV and runs:

```text
ingest -> clean -> validate -> save processed data -> train
```

If validation fails, training is skipped.

## Tests

The project includes a small pytest suite focused on the most important behavior.

`test_validate.py` checks that:

- valid data passes validation
- invalid data fails validation

`test_pipeline.py` checks that:

- model training is not called when validation fails

Run all tests with:

```bash
pytest -v
```

## Continuous Integration

A GitHub Actions workflow is included in:

```text
.github/workflows/ci.yml
```

The workflow runs on pushes and pull requests.

It:

```text
checks out the repository
        |
        v
sets up Python
        |
        v
installs requirements
        |
        v
runs pytest
```

This provides a simple automated check that the validation and pipeline tests continue to pass after code changes.

## Setup

Clone the repository:

```bash
git clone https://github.com/SarathKumarKomathukattil/ml-data-pipeline-churn.git
cd ml-data-pipeline-churn
```

Create and activate a virtual environment if required, then install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
scikit-learn
joblib
pytest
```

## Assumptions

For this take-home implementation, the generated daily CSV contains the `churn` label so that the full workflow can be demonstrated from ingestion through model training.

The trained model currently replaces the existing `churn_pipeline.joblib` file on each successful run. Model versioning is also outside the scope of this implementation.

## Summary

The main goal of the project is to demonstrate a simple and clear conditional ML workflow:

```text
daily raw CSV
      |
      v
ingestion
      |
      v
cleaning
      |
      v
data-quality validation
      |
      +---- fail ----> stop
      |
     pass
      |
      v
processed dataset
      |
      v
model training
```

The key behavior is that **model training only happens when the incoming data passes the defined quality checks**.