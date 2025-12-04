import streamlit as st
import pandas as pd
from autoclasp import issues as issues_module
from autoclasp import preprocessing as preprocessing_module


def _small_circle(value: int):
    """
    Render a compact circular percentage indicator with HTML/CSS.
    Shows current percentage in the center with blue color scheme.
    """
    pct = max(0, min(100, int(value)))
    
    # Determine color based on health score
    if pct >= 80:
        color = "#2196F3"  # Blue for good health
    elif pct >= 60:
        color = "#42A5F5"  # Light blue for moderate
    else:
        color = "#64B5F6"  # Lighter blue for poor health
    
    # CSS for a polished radial progress visualization
    html = f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:16px;padding:12px;">
      <div style="position:relative;width:100px;height:100px;">
        <svg viewBox="0 0 36 36" width="100" height="100" style="transform:rotate(-90deg);">
          <path d="M18 2.0845
                   a 15.9155 15.9155 0 0 1 0 31.831
                   a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none" stroke="#E3F2FD" stroke-width="2.5"/>
          <path d="M18 2.0845
                   a 15.9155 15.9155 0 0 1 0 31.831
                   a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none" stroke="{color}" stroke-width="2.5"
                stroke-dasharray="{pct}, 100" stroke-linecap="round"
                style="transition: stroke-dasharray 0.3s ease;"/>
        </svg>
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
          <div style="font-size:20px;font-weight:600;color:#1976D2;">{pct}%</div>
          <div style="font-size:11px;color:#666;margin-top:2px;">Health</div>
        </div>
      </div>
      <div style="font-size:16px;font-weight:500;color:#1976D2;">Data Health Score</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def _has_selected_actions(decisions: dict) -> bool:
    """
    Returns True if at least one preprocessing action is selected by the user.
    Train-test split is not considered an action here.
    """
    if not decisions:
        return False

    # Actions with explicit "apply" flag
    apply_flags = [
        decisions.get("duplicates", {}).get("apply", False),
        decisions.get("constant_features", {}).get("apply", False),
        decisions.get("missing_values", {}).get("apply", False),
        decisions.get("outliers", {}).get("apply", False),
    ]

    # Encoding/scaling considered actions when not "none"
    encoding = decisions.get("encoding", "none")
    scaling = decisions.get("scaling", "none")
    has_encoding = encoding in ("onehot", "ordinal")
    has_scaling = scaling in ("standard", "minmax")

    return any(apply_flags) or has_encoding or has_scaling

def main():
    # Stage 3 - Issue Detection only
    st.title("Stage 3: Issue Detection")
    # Check dataset availability
    if st.session_state.get("dataframe") is None:
        st.error("No dataset found. Please upload a dataset in Stage 1.")
        if st.button("Go to Stage 1"):
            st.switch_page("pages/upload_and_eda.py")
        return

    # Get dataframe and target column from session
    df = st.session_state.dataframe
    target_col = st.session_state.target_column

    # Ensure decisions container
    if "user_decisions" not in st.session_state:
        st.session_state.user_decisions = {}

    # a) Show data shape and target column
    stats = pd.DataFrame(
        [{"Metric": "Rows", "Count": df.shape[0]},
         {"Metric": "Columns", "Count": df.shape[1]}]
    )
    st.markdown("### a) Data Shape")
    st.write(stats)
    st.write(f"Target column selected: {target_col}")

    # Stage 3.1: Data Quality Analysis (detection only)
    st.markdown("---")
    st.markdown("#### Stage 3.1: Data Quality Analysis")

    # Detect all issues and store in session
    detected = issues_module.detect_all_issues(df, target_col)
    st.session_state.detected_issues = detected

    # Health score: metric + compact circular indicator
    health = detected["health_score"]
    col_h1, col_h2 = st.columns([1, 3])
    with col_h1:
        st.metric("Data Health", f"{health}/100")
    with col_h2:
        _small_circle(health)

    # Score breakdown table showing per-issue deduction and status
    breakdown_rows = []
    for issue_name, pts in detected["score_breakdown"].items():
        breakdown_rows.append({
            "Issue": issue_name,
            "Deduction": pts if pts != 0 else 0,
            "Status": "OK" if pts == 0 else "Affected"
        })
    breakdown_df = pd.DataFrame(breakdown_rows)
    st.markdown("Score breakdown")
    st.dataframe(breakdown_df, use_container_width=True)

    # Stage 3.2: Detected Issues (report only)
    st.markdown("---")
    st.markdown("#### Stage 3.2: Detected Issues")

    # Missing values
    st.markdown("### a) Missing Values")
    mv = detected["missing_values"]
    if mv["has_issue"]:
        st.warning(f"Missing values found in {mv['affected_columns']} columns")
        mv_df = pd.DataFrame([{"Column": c, "Missing %": f"{p:.2f}%"} for c, p in mv["details"].items()])
        st.dataframe(mv_df, use_container_width=True)
    else:
        st.success("No missing values detected")

    # Outliers
    st.markdown("### b) Outliers")
    ol = detected["outliers"]
    if ol["has_issue"]:
        st.warning(f"Outliers detected in {ol['affected_columns']} columns")
        ol_df = pd.DataFrame([{"Column": c, "Outlier Count": cnt} for c, cnt in ol["details"].items()])
        st.dataframe(ol_df, use_container_width=True)
    else:
        st.success("No significant outliers detected")

    # Class imbalance
    st.markdown("### c) Class Imbalance")
    imb = detected["class_imbalance"]
    if imb["has_issue"]:
        st.warning(f"Class imbalance detected (minority/majority ratio: {imb['ratio']:.2f})")
        imb_df = pd.DataFrame([{"Class": k, "Count": v} for k, v in imb["distribution"].items()])
        st.dataframe(imb_df, use_container_width=True)
    else:
        st.success("Classes are reasonably balanced")

    # High cardinality
    st.markdown("### d) High Cardinality Categorical Features")
    hc = detected["high_cardinality"]
    if hc["has_issue"]:
        st.warning(f"High cardinality detected in {len(hc['columns'])} categorical columns")
        hc_df = pd.DataFrame([{"Column": c, "Unique Values": u} for c, u in hc["details"].items()])
        st.dataframe(hc_df, use_container_width=True)
    else:
        st.success("No high cardinality categorical features")

    # Constant features
    st.markdown("### e) Constant or Near-Constant Features")
    cf = detected["constant_features"]
    if cf["has_issue"]:
        st.warning(f"Constant or near-constant features found: {cf['count']} column(s)")
        st.write(", ".join(cf["columns"]))
    else:
        st.success("No constant features detected")

    # Duplicates
    st.markdown("### f) Duplicate Rows")
    dup = detected["duplicates"]
    if dup["has_issue"]:
        st.warning(f"Duplicate rows detected: {dup['count']} ({dup['percentage']:.2f}%)")
    else:
        st.success("No duplicate rows detected")

    # Stage 4 - Preprocessing Configuration (suggestions and apply)
    st.markdown("---")
    st.markdown("#### Stage 4: Preprocessing Configuration")

    # Decisions store and detected reference
    decisions = st.session_state.user_decisions
    detected = st.session_state.detected_issues

    # Missing values configuration
    st.markdown("##### Missing Values")
    mv_method = st.selectbox(
        "Imputation method",
        options=["median", "mean", "mode", "constant", "drop_rows", "drop_columns"],
        key="mv_method",
    )
    mv_const_val = st.text_input("Constant value (used when method = constant)", key="mv_const_val") if mv_method == "constant" else None
    mv_apply = st.checkbox("Apply missing values handling", key="mv_apply", value=False)
    decisions["missing_values"] = {"apply": mv_apply, "method": mv_method, "constant_value": mv_const_val}

    # Outliers configuration
    st.markdown("##### Outliers")
    ol_method = st.selectbox("Outlier handling", options=["cap", "remove", "none"], key="ol_method")
    ol_apply = st.checkbox("Apply outlier handling", key="ol_apply", value=False)
    ol_columns = list(detected.get("outliers", {}).get("details", {}).keys())
    decisions["outliers"] = {"apply": ol_apply, "method": ol_method, "columns": ol_columns}

    # Constant features configuration
    st.markdown("##### Constant/Near-Constant Features")
    cf_columns = detected.get("constant_features", {}).get("columns", [])
    cf_apply = st.checkbox("Remove constant features", key="cf_apply", value=False)
    decisions["constant_features"] = {"apply": cf_apply, "columns": cf_columns}

    # Duplicates configuration
    st.markdown("##### Duplicates")
    dup_apply = st.checkbox("Remove duplicate rows", key="dup_apply", value=False)
    decisions["duplicates"] = {"apply": dup_apply}

    # Encoding
    st.markdown("##### Encoding")
    enc_options = ["onehot", "ordinal", "none"]
    enc_default = st.session_state.user_decisions.get("encoding", "none")
    encoding = st.selectbox(
        "Categorical encoding",
        options=enc_options,
        index=enc_options.index(enc_default),  # default to "none"
        key="encoding_method_ui",
    )
    decisions["encoding"] = encoding

    # Scaling
    st.markdown("##### Scaling")
    sc_options = ["none", "standard", "minmax"]
    sc_default = st.session_state.user_decisions.get("scaling", "none")
    scaling = st.selectbox(
        "Feature scaling",
        options=sc_options,
        index=sc_options.index(sc_default),   # default to "none"
        key="scaling_method_ui",
    )
    decisions["scaling"] = scaling

    # Train-test split
    st.markdown("##### Train-Test Split")
    default_ts = float(st.session_state.get("train_size", 0.8))
    train_size = st.slider("Train size", 0.5, 0.95, default_ts, 0.05, key="train_size_ui")
    st.session_state.train_size = train_size

    # Stage 5 - Apply preprocessing
    st.markdown("---")
    st.markdown("#### Stage 5: Apply Preprocessing")

    if st.button("Apply preprocessing"):
        # Validate that the user selected at least one preprocessing action
        if not _has_selected_actions(st.session_state.user_decisions):
            st.warning("Please select at least one preprocessing option (e.g., missing values, outliers, duplicates, constant features, encoding, or scaling) before applying.")
            # Clear previous preview to avoid confusion
            st.session_state.preprocessing_applied = False
            st.session_state.df_clean = None
        else:
            with st.spinner("Processing..."):
                df_clean, X_train, X_test, y_train, y_test = preprocessing_module.apply_preprocessing(
                    df=st.session_state.dataframe,
                    target_col=st.session_state.target_column,
                    decisions=st.session_state.user_decisions,
                    train_size=st.session_state.train_size,
                )
                st.session_state.df_clean = df_clean
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.preprocessing_applied = True

    # Show preview and download button once preprocessing is applied
    if st.session_state.get("preprocessing_applied"):
        st.markdown("Cleaned data (before split)")
        st.dataframe(st.session_state.df_clean.head(), use_container_width=True)

        # Download cleaned CSV
        csv_bytes = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

        st.markdown("Train/Test shapes")
        shape_df = pd.DataFrame(
            [
                {"Split": "X_train", "Rows": len(st.session_state.X_train), "Cols": st.session_state.X_train.shape[1]},
                {"Split": "X_test", "Rows": len(st.session_state.X_test), "Cols": st.session_state.X_test.shape[1]},
                {"Split": "y_train", "Rows": len(st.session_state.y_train), "Cols": 1},
                {"Split": "y_test", "Rows": len(st.session_state.y_test), "Cols": 1},
            ]
        )
        st.dataframe(shape_df, use_container_width=True)


if __name__ == "__main__":
    main()





