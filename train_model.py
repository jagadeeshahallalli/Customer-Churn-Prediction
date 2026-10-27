
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    # CustomerID is an identifier, not a useful predictive feature.
    df = df.drop(columns=["customerID"], errors="ignore")

    # The source stores TotalCharges as text; convert invalid blanks to NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Normalize the target into numeric labels.
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    if df["Churn"].isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    return df


def build_pipeline(model):
    df_columns = None
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    # Remaining input columns are categorical.
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def main():
    print("Loading customer churn dataset...")
    df = load_data()

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }

        results.append(metrics)
        fitted_models[name] = pipeline

        print(pd.Series(metrics).to_string())
        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ))

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Score", ascending=False
    )

    results_df.to_csv(
        os.path.join(REPORT_DIR, "model_comparison.csv"),
        index=False,
    )

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    joblib.dump(best_model, os.path.join(MODEL_DIR, "churn_model.joblib"))

    # Save test labels and probabilities for dashboard evaluation charts.
    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    evaluation = {
        "best_model": best_name,
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, best_predictions)),
        "precision": float(precision_score(
            y_test, best_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, best_predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, best_predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, best_probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, best_predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            best_predictions,
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    with open(
        os.path.join(REPORT_DIR, "evaluation.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(evaluation, file, indent=4)

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print(f"\nBest model by F1 Score: {best_name}")
    print("Saved model: models/churn_model.joblib")
    print("Saved comparison: reports/model_comparison.csv")
    print("Saved evaluation: reports/evaluation.json")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
