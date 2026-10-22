import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

def train_model(df):
    X = df.drop(columns = ['user_id','snapshot_datetime','churn'])
    y = df['churn']

    numeric_features = [
        "age",
        "tenure_months",
        "avg_monthly_spend",
        "avg_session_minutes",
        "support_tickets",
        "days_since_last_login",
    ]

    categorical_features = ['plan_type']

    preprocessor = ColumnTransformer(transformers=[
        ('numeric',StandardScaler(),numeric_features),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ])

    X_train,X_test,y_train,y_test = train_test_split(X,
                                                     y,
                                                     test_size=0.2,
                                                     random_state=42,
                                                     stratify=y)

    model = Pipeline(
        steps=[
            ('preprocessor',preprocessor),
            ('classifier',LogisticRegression(max_iter=1000))
        ]
    )
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:,1]

    print('\nModel training completed')
    print(f'Accuracy: {accuracy_score(y_test,predictions):.3f}')
    print(f'Recall: {recall_score(y_test,predictions):.3f}')
    print(f'Precision: {precision_score(y_test,predictions):.3f}')
    print(f'F1 Score: {f1_score(y_test,predictions):.3f}')
    print(f'ROC-AUC: {roc_auc_score(y_test,probabilities):.3f}')

    model_dir = Path('models')
    model_dir.mkdir(parents=True,exist_ok=True)

    model_path = model_dir/'churn_pipeline.joblib'
    joblib.dump(model,model_path)

    print(f'\nModel saved to {model_path}')

