import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)

    # Fix TotalCharges (has spaces as missing values)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customerID (not predictive)
    df.drop("customerID", axis=1, inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode binary yes/no columns
    binary_cols = [c for c in df.columns
                   if df[c].nunique() == 2 and df[c].dtype == object]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, drop_first=True)

    return df

def get_train_test(path=DATA_PATH, test_size=0.2, random_state=42):
    df = load_and_preprocess(path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # Save scaler + feature names for the app
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    return X_train, X_test, y_train, y_test, X.columns.tolist()