import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Function to drop duplicate rows when approved
def _drop_duplicates(df: pd.DataFrame, apply: bool) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True) if apply else df


# Function to drop constant columns when approved
def _drop_constant(df: pd.DataFrame, columns: list | None) -> pd.DataFrame:
    if columns:
        cols = [c for c in columns if c in df.columns]
        if cols:
            return df.drop(columns=cols)
    return df


# Function to handle missing values with selected method
def _impute_missing(df: pd.DataFrame, method: str, constant_value: str | None = None) -> pd.DataFrame:
    # Drop rows or columns containing NaN if chosen
    if method == "drop_rows":
        return df.dropna()
    if method == "drop_columns":
        cols = [c for c in df.columns if df[c].isna().any()]
        return df.drop(columns=cols) if cols else df

    # Fill NaN values depending on method
    dfc = df.copy()
    num_cols = dfc.select_dtypes(include=[np.number]).columns
    cat_cols = dfc.select_dtypes(exclude=[np.number]).columns

    if method == "mean":
        if len(num_cols) > 0:
            dfc[num_cols] = dfc[num_cols].fillna(dfc[num_cols].mean())
        for c in cat_cols:
            if dfc[c].isna().any():
                dfc[c] = dfc[c].fillna(dfc[c].mode().iloc[0])
    elif method == "median":
        if len(num_cols) > 0:
            dfc[num_cols] = dfc[num_cols].fillna(dfc[num_cols].median())
        for c in cat_cols:
            if dfc[c].isna().any():
                dfc[c] = dfc[c].fillna(dfc[c].mode().iloc[0])
    elif method == "mode":
        for c in dfc.columns:
            if dfc[c].isna().any():
                dfc[c] = dfc[c].fillna(dfc[c].mode().iloc[0])
    elif method == "constant":
        # Use a single constant value for all columns
        for c in dfc.columns:
            if dfc[c].isna().any():
                if pd.api.types.is_numeric_dtype(dfc[c]):
                    try:
                        v = float(constant_value) if constant_value is not None else 0.0
                    except Exception:
                        v = 0.0
                    dfc[c] = dfc[c].fillna(v)
                else:
                    v = str(constant_value) if constant_value is not None else "missing"
                    dfc[c] = dfc[c].fillna(v)
    return dfc


# Helper to compute IQR bounds for outlier handling
def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return low, high


# Function to handle outliers on selected columns with chosen method
def _handle_outliers(df: pd.DataFrame, columns: list | None, method: str) -> pd.DataFrame:
    # No action if no columns or method is none
    if not columns or method == "none":
        return df

    dfc = df.copy()
    num_cols = set(dfc.select_dtypes(include=[np.number]).columns)

    # Filter to numeric columns present in df
    cols = [c for c in (columns or []) if c in num_cols]
    if not cols:
        return df

    if method == "cap":
        # Clip values to IQR-based bounds
        for c in cols:
            low, high = _iqr_bounds(dfc[c].dropna())
            dfc[c] = dfc[c].clip(lower=low, upper=high)
        return dfc

    if method == "remove":
        # Remove rows containing outliers
        mask = pd.Series(False, index=dfc.index)
        for c in cols:
            s = dfc[c]
            low, high = _iqr_bounds(s.dropna())
            mask = mask | (s < low) | (s > high)
        return dfc.loc[~mask].reset_index(drop=True)

    return df


# Function to encode categorical variables using one-hot (train/test aligned)
def _encode_onehot_train_test(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = pd.get_dummies(X_tr, drop_first=False, dtype=np.int8)
    Xte = pd.get_dummies(X_te, drop_first=False, dtype=np.int8)
    # Align test columns to train columns (fill missing with zeros)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte


# Function to encode categorical variables using ordinal mapping (train categories define mapping)
def _encode_ordinal_train_test(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = X_tr.copy()
    Xte = X_te.copy()
    cat_cols = Xtr.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        categories = Xtr[c].astype("category").cat.categories
        mapping = {k: i for i, k in enumerate(categories)}
        Xtr[c] = Xtr[c].map(mapping).fillna(-1).astype(np.int32)
        Xte[c] = Xte[c].map(mapping).fillna(-1).astype(np.int32)
    return Xtr, Xte


# Function to scale numeric columns using StandardScaler or MinMaxScaler
def _scale_numeric_train_test(X_tr: pd.DataFrame, X_te: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # No scaling when method is none
    if method not in ("standard", "minmax"):
        return X_tr, X_te

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    Xtr = X_tr.copy()
    Xte = X_te.copy()

    # Select numeric columns
    num_cols = Xtr.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return Xtr, Xte

    # Fit on train, transform train and test
    scaler.fit(Xtr[num_cols])
    Xtr[num_cols] = scaler.transform(Xtr[num_cols])
    Xte[num_cols] = scaler.transform(Xte[num_cols])
    return Xtr, Xte


# Main function to apply preprocessing based on user decisions
def apply_preprocessing(
    df: pd.DataFrame,
    target_col: str,
    decisions: dict,
    train_size: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Working copy
    dfc = df.copy()

    # 1) Duplicate rows removal (optional)
    dup_apply = bool(decisions.get("duplicates", {}).get("apply", False))
    dfc = _drop_duplicates(dfc, dup_apply)

    # 2) Constant feature removal (optional)
    const = decisions.get("constant_features", {})
    dfc = _drop_constant(dfc, const.get("columns") if const.get("apply") else None)

    # 3) Missing values handling (optional)
    mv = decisions.get("missing_values", {})
    if mv.get("apply"):
        dfc = _impute_missing(dfc, mv.get("method", "median"), mv.get("constant_value"))

    # 4) Outlier handling (optional)
    ol = decisions.get("outliers", {})
    if ol.get("apply"):
        dfc = _handle_outliers(dfc, ol.get("columns", []), ol.get("method", "none"))

    # 5) Train/test split with optional stratification
    y = dfc[target_col]
    X = dfc.drop(columns=[target_col])

    # Guard: stratify only if every class has at least 2 samples and there are >=2 classes
    counts = y.value_counts(dropna=False)
    can_stratify = (counts.size >= 2) and (counts.min() >= 2)
    stratify = y if can_stratify else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        # Fallback: non-stratified split if scikit-learn still complains
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=None
        )

    # 6) Encoding choice (one-hot / ordinal / none)
    enc_method = decisions.get("encoding", "onehot")
    if enc_method == "onehot":
        X_train, X_test = _encode_onehot_train_test(X_train, X_test)
    elif enc_method == "ordinal":
        X_train, X_test = _encode_ordinal_train_test(X_train, X_test)

    # 7) Scaling choice (standard / minmax / none)
    sc_method = decisions.get("scaling", "none")
    X_train, X_test = _scale_numeric_train_test(X_train, X_test, sc_method)

    # Return cleaned data and splits
    return dfc, X_train, X_test, y_train, y_test