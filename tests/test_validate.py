import pandas as pd
from validate import validate_data


def get_valid_data():
    return pd.DataFrame({
        "user_id": ["USER_00001", "USER_00002"],
        "snapshot_datetime": [
            "2026-10-22T10:00:00Z",
            "2026-10-22T10:00:00Z"
        ],
        "age": [30, 45],
        "tenure_months": [12, 24],
        "avg_monthly_spend": [100.0, 150.0],
        "avg_session_minutes": [25, 40],
        "support_tickets": [1, 0],
        "days_since_last_login": [2, 5],
        "plan_type": ["Basic", "Premium"],
        "churn": [0, 1]
    })


def test_valid_data_passes():
    df = get_valid_data()
    is_valid, errors = validate_data(df)

    assert is_valid is True
    assert errors == []


def test_invalid_age_fails():
    df = get_valid_data()
    df.loc[0, "age"] = 10
    is_valid, errors = validate_data(df)

    assert is_valid is False