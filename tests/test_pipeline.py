import pipeline
import pandas as pd

def test_training_skipped_when_validation_fails(tmp_path,monkeypatch):
    df = pd.DataFrame({
        "user_id": ["USER_00001"],
        "snapshot_datetime": ["2026-10-22T10:00:00Z"],
        "age": [10],   # invalid age is entered 
        "tenure_months": [12],
        "avg_monthly_spend": [100.0],
        "avg_session_minutes": [25],
        "support_tickets": [1],
        "days_since_last_login": [2],
        "plan_type": ["Basic"],
        "churn": [0]
    })

    bad_file = tmp_path/'bad_data.csv'

    df.to_csv(bad_file,
              index= False)

    training_called = False
    def fake_train_model(df):
        nonlocal training_called
        training_called = True


    monkeypatch.setattr(
        pipeline,
        'train_model',
        fake_train_model
    )

    pipeline.run_pipeline(bad_file)
    assert training_called is False

