"""
Trains the Random Forest heart disease risk model from scratch and saves
the artifacts used by app.py. Run this to reproduce model/heart_disease_model.pkl,
model/scaler.pkl, model/feature_columns.pkl and model/feature_importances.csv.

Usage:
    python train_model.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]


def main():
    df = pd.read_csv("data/heart_disease_uci_original.csv")

    # Binary target: 0 = no disease, 1 = disease present (collapses severity 1-4)
    df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop("num", axis=1).dropna()
    df.to_csv("data/heart_disease_processed.csv", index=False)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"AUC:      {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred))

    joblib.dump(model, "model/heart_disease_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(list(X.columns), "model/feature_columns.pkl")

    importances = pd.DataFrame({
        "feature": X.columns, "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv("model/feature_importances.csv", index=False)

    print("\nSaved model, scaler, feature_columns and feature_importances to model/")


if __name__ == "__main__":
    main()
