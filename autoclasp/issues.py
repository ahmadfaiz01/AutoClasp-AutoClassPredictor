import pandas as pd
import numpy as np

from autoclasp import eda as eda_module


# Function to detect missing values in the dataframe
def detect_missing_values(df: pd.DataFrame) -> dict:
    # Count missing values per column
    total_missing = df.isnull().sum()
    # Compute missing percentage per column (protect against empty df)
    percent_missing = (total_missing / len(df)) * 100 if len(df) > 0 else 0
    # Filter columns that have any missing values
    cols = percent_missing[percent_missing > 0]
    # Build details mapping: column -> missing percentage (float)
    details = {col: float(percent_missing[col]) for col in cols.index}
    # Return structured detection info
    return {
        "has_issue": len(cols) > 0,
        "affected_columns": len(cols),
        "details": details,
    }


# Function to detect outliers using IQR results from EDA
def detect_outliers(df: pd.DataFrame, threshold_pct: float = 5.0) -> dict:
    # Reuse EDA outlier summary (expected columns: column, outlier_count, percentage)
    iqr_df = eda_module.detect_outliers_iqr(df)
    if iqr_df is None or iqr_df.empty:
        return {"has_issue": False, "affected_columns": 0, "details": {}}
    # Keep columns above a percentage threshold
    problematic = iqr_df[iqr_df["percentage"] > threshold_pct]
    # Build details mapping: column -> outlier count
    details = {row["column"]: int(row["outlier_count"]) for _, row in problematic.iterrows()}
    # Return structured detection info
    return {
        "has_issue": len(problematic) > 0,
        "affected_columns": len(problematic),
        "details": details,
        "threshold_pct": threshold_pct,
    }


# Function to detect class imbalance in target column
def detect_class_imbalance(df: pd.DataFrame, target_col: str, ratio_threshold: float = 0.3) -> dict:
    # Validate target column existence
    if target_col not in df.columns:
        return {"has_issue": False, "ratio": 1.0, "distribution": {}}
    # Compute class distribution (including NaN)
    counts = df[target_col].value_counts(dropna=False)
    if counts.empty:
        return {"has_issue": False, "ratio": 1.0, "distribution": {}}
    # Compute minority/majority ratio (avoid division by zero)
    min_c, max_c = counts.min(), counts.max()
    ratio = float(min_c / max_c) if max_c > 0 else 1.0
    # Serialize distribution to dict (JSON-friendly keys)
    distribution = {str(k): int(v) for k, v in counts.items()}
    # Return structured detection info
    return {
        "has_issue": ratio < ratio_threshold,
        "ratio": ratio,
        "distribution": distribution,
        "ratio_threshold": ratio_threshold,
    }


# Function to detect high cardinality categorical features
def detect_high_cardinality(df: pd.DataFrame, unique_threshold: int = 50) -> dict:
    # Select categorical columns (object, category, bool)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    # Collect columns exceeding unique value threshold
    details = {}
    for col in cat_cols:
        u = int(df[col].nunique(dropna=True))
        if u > unique_threshold:
            details[col] = u
    # Return structured detection info
    return {
        "has_issue": len(details) > 0,
        "columns": list(details.keys()),
        "details": details,
        "unique_threshold": unique_threshold,
    }


# Function to detect constant or near-constant features
def detect_constant_features(df: pd.DataFrame) -> dict:
    # Columns with <= 1 unique value (excluding NaN)
    cols = []
    for col in df.columns:
        u = df[col].nunique(dropna=True)
        if u <= 1:
            cols.append(col)
    # Return structured detection info
    return {
        "has_issue": len(cols) > 0,
        "columns": cols,
        "count": len(cols),
    }


# Function to detect duplicate rows
def detect_duplicates(df: pd.DataFrame, pct_threshold: float = 1.0) -> dict:
    # Duplicate rows count (excluding first occurrence)
    dup_count = int(df.duplicated().sum())
    # Percentage of duplicates (protect against empty df)
    total = len(df)
    pct = float((dup_count / total) * 100) if total > 0 else 0.0
    # Return structured detection info
    return {
        "has_issue": pct > pct_threshold,
        "count": dup_count,
        "percentage": pct,
        "pct_threshold": pct_threshold,
    }


# Helper to calculate overall health score and per-issue deductions
def _calculate_health_score(missing_info, outlier_info, imb_info, card_info, const_info, dup_info) -> tuple[int, dict]:
    # Start at 100 and subtract issue-specific penalties
    score = 100
    breakdown = {}

    # Missing values: subtract 5 per affected column (cap at -25)
    if missing_info["has_issue"]:
        d = min(missing_info["affected_columns"] * 5, 25)
        score -= d
        breakdown["Missing Values"] = -d
    else:
        breakdown["Missing Values"] = 0

    # Outliers: subtract 3 per affected column (cap at -15)
    if outlier_info["has_issue"]:
        d = min(outlier_info["affected_columns"] * 3, 15)
        score -= d
        breakdown["Outliers"] = -d
    else:
        breakdown["Outliers"] = 0

    # Class imbalance: subtract more when ratio is worse
    # ratio < 0.1 -> -20, < 0.2 -> -15, otherwise -10
    if imb_info["has_issue"]:
        r = imb_info["ratio"]
        d = 20 if r < 0.1 else 15 if r < 0.2 else 10
        score -= d
        breakdown["Class Imbalance"] = -d
    else:
        breakdown["Class Imbalance"] = 0

    # High cardinality: subtract 5 per high-card column (cap at -15)
    if card_info["has_issue"]:
        d = min(len(card_info["columns"]) * 5, 15)
        score -= d
        breakdown["High Cardinality"] = -d
    else:
        breakdown["High Cardinality"] = 0

    # Constant features: subtract 3 per constant column (cap at -10)
    if const_info["has_issue"]:
        d = min(const_info["count"] * 3, 10)
        score -= d
        breakdown["Constant Features"] = -d
    else:
        breakdown["Constant Features"] = 0

    # Duplicates: subtract by percentage severity
    # >10% -> -15, >5% -> -10, else -> -5
    if dup_info["has_issue"]:
        p = dup_info["percentage"]
        d = 15 if p > 10 else 10 if p > 5 else 5
        score -= d
        breakdown["Duplicates"] = -d
    else:
        breakdown["Duplicates"] = 0

    # Clamp final score to [0, 100]
    score = max(0, score)
    return score, breakdown


# Function to run all detections and return a single dictionary
def detect_all_issues(df: pd.DataFrame, target_col: str) -> dict:
    # Run individual detectors
    missing_info = detect_missing_values(df)
    outlier_info = detect_outliers(df)
    imb_info = detect_class_imbalance(df, target_col)
    card_info = detect_high_cardinality(df)
    const_info = detect_constant_features(df)
    dup_info = detect_duplicates(df)
    # Compute overall health score and per-issue deductions
    health_score, breakdown = _calculate_health_score(
        missing_info, outlier_info, imb_info, card_info, const_info, dup_info
    )
    # Pack all results
    return {
        "missing_values": missing_info,
        "outliers": outlier_info,
        "class_imbalance": imb_info,
        "high_cardinality": card_info,
        "constant_features": const_info,
        "duplicates": dup_info,
        "health_score": health_score,
        "score_breakdown": breakdown,
    }