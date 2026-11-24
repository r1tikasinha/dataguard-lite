import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def run_quality_audit(df: pd.DataFrame, contamination: float = 0.03):
    clean_df = df.copy()

    # 1. Basic Data Engineering Health Checks
    total_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    missing_cells = int(df.isnull().sum().sum())
    missing_counts = df.isnull().sum()
    missing_by_col = missing_counts[missing_counts > 0].to_dict()

    # 2. Unsupervised ML: Isolation Forest on numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    anomalies_df = pd.DataFrame()

    # Rule-based check: negative values in columns that look like quantities/prices/amounts
    negative_by_col = {c: int((df[c] < 0).sum()) for c in numeric_cols if (df[c] < 0).any()}

    if len(numeric_cols) >= 1 and total_rows >= 10:
        # Fill missing values temporarily with median just for ML scoring
        ml_input = df[numeric_cols].fillna(df[numeric_cols].median()).fillna(0)

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        preds = iso_forest.fit_predict(ml_input)
        scores = iso_forest.decision_function(ml_input)

        df_scored = df.copy()
        df_scored["Anomaly_Score"] = np.round(scores, 4)
        df_scored["Status"] = np.where(preds == -1, "Anomaly", "Normal")

        anomalies_df = df_scored[df_scored["Status"] == "Anomaly"].copy()
    else:
        df_scored = df.copy()
        df_scored["Status"] = "Normal"
        df_scored["Anomaly_Score"] = 0.0

    # 3. Auto-Cleaned Preview Dataset
    clean_df = clean_df.drop_duplicates()
    if len(anomalies_df) > 0:
        clean_df = clean_df.drop(index=anomalies_df.index, errors="ignore")
    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    for col in clean_df.select_dtypes(include=["object"]).columns:
        clean_df[col] = clean_df[col].fillna("UNKNOWN")

    # Health Score Calculation (0 to 100)
    issue_rows = duplicate_count + len(anomalies_df) + int(df.isnull().any(axis=1).sum())
    health_score = max(0, round(100 - (issue_rows / max(total_rows, 1)) * 100, 1))

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_count,
        "missing_cells": missing_cells,
        "missing_by_col": missing_by_col,
        "negative_by_col": negative_by_col,
        "anomaly_count": len(anomalies_df),
        "numeric_cols": numeric_cols,
        "health_score": health_score,
        "scored_df": df_scored,
        "anomalies_df": anomalies_df,
        "clean_df": clean_df,
    }
