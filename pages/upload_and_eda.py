import streamlit as st
import pandas as pd

# Importing EDA related utilities
from autoclasp import eda as eda_module


def ensure_session_state_defaults():
    defaults = {
        "dataframe": None,
        "df": None,
        "processed_df": None,
        "uploaded_file": None,
        "target_column": None,
        "train_size": 0.8,
        "user_decisions": {},
        "preprocessing_applied": False,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "trained_models": {},
        "evaluation_metrics": None,
        "last_viewed_comparison": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    ensure_session_state_defaults()

    # Stage 1 - Dataset Uploading and Basic Information
    st.title("Stage 1: Dataset Uploading and Basic Information")

    # Asking User to upload a file
    st.markdown("### Stage 1.1: Upload Your Dataset")

    # Check if data already exists in session
    if st.session_state.dataframe is not None:
        st.success("Dataset already loaded!")

        # Give option to upload new dataset
        if st.button("Upload New Dataset"):
            st.session_state.uploaded_file = None
            st.session_state.dataframe = None
            st.session_state.df = None
            st.session_state.target_column = None
            st.session_state.processed_df = None
            st.rerun()
    else:
        # Handling the input of the dataset
        file_uploaded = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx", "xls"],
            help="CSV, XLSX, XLS supported formats only"
        )

        # If a file is uploaded, check the type and read it into a DataFrame
        if file_uploaded is not None:
            try:
                if file_uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_uploaded)
                else:
                    df = pd.read_csv(file_uploaded)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return

            # Preprocessing data for safer EDA
            processed_df = eda_module.clean_and_coerce(df, threshold=0.5)
            non_numeric_report = eda_module.report_non_numeric(df)

            # Storing the uploaded file and dataframe in session state
            st.session_state.uploaded_file = file_uploaded
            st.session_state.processed_df = processed_df
            st.session_state.dataframe = df
            st.session_state.df = df  # This is what final_report.py looks for

            # Reporting non-numeric columns if any
            if not non_numeric_report.empty:
                st.warning(
                    "Some columns contained non-numeric values; they were detected and converted where possible."
                )
                st.dataframe(non_numeric_report)

            st.success("File uploaded successfully!")
            st.rerun()  # Reload to show the data instead of uploader

    # Only show data analysis if dataframe exists
    if st.session_state.dataframe is not None:
        df_original = st.session_state.dataframe

        # Using processed dataframe for EDA
        if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
            df = st.session_state.processed_df
        else:
            df = df_original

        # A2 - Basic Data Information
        st.markdown("### Stage 1.2: Basic Data Information")

        st.write("Here is a preview of your dataset:")
        st.dataframe(df.head(5))  # Displaying first 5 rows of the dataframe

        # No. of Rows and Columns
        stats = [
            {
                "Metric": "Rows",
                "Count": df.shape[0]
            },
            {
                "Metric": "Columns",
                "Count": df.shape[1]
            }
        ]
        data_shape = pd.DataFrame(stats)
        st.markdown("### a) Data Shape")
        st.write(data_shape)

        # Feature Names and Datatypes
        st.write("### b) Feature Names and Data Types")

        datatype_info = []
        for col in df.columns:
            datatype_info.append({
                "Column": col,
                "Data Type": df[col].dtype
            })
        datatype_df = pd.DataFrame(datatype_info)
        st.write(datatype_df)

        # Data Statistics
        st.write("### c) Data Statistics")
        st.write(df.describe(include='all').T)  # caters to both numerical and categorical data

        # Checking Class Distribution (whether balanced or not)
        st.write("### d) Class Distribution")

        # Highlighting the target selection box
        st.markdown(
            "<div style='background:#800000; padding:8px; border-radius:6px; border-left:4px; margin-bottom:8px;'>"
            "<strong>Please select target column!</strong>"
            "</div>",
            unsafe_allow_html=True
        )

        # Asking for the target value column
        cols = df.columns.tolist()
        default_index = 0
        if st.session_state.target_column in cols:
            try:
                default_index = cols.index(st.session_state.target_column)
            except Exception:
                default_index = 0

        target_col = st.selectbox(
            "Select the target column for class distribution",
            options=cols,
            index=default_index
        )
        st.session_state.target_column = target_col  # Updating target column in session state

        # Calling the show class distribution function
        try:
            eda_module.show_class_distribution(df, target_col)
        except Exception as e:
            st.error(f"Could not render class distribution: {e}")

        # Stage 2 - EDA
        st.title("Stage 2: Exploratory Data Analysis (EDA)")

        # Missing value analysis
        st.markdown("#### Stage 2.1: Missing Value Analysis")
        try:
            eda_module.show_missing_values(df)
        except Exception as e:
            st.error(f"Missing value analysis failed: {e}")

        # B.2: Outlier detection (placeholder summary for now)
        st.markdown("#### Stage 2.2 Outlier Detection Summary")
        try:
            eda_module.show_outlier_summary(df)
        except Exception as e:
            st.error(f"Outlier summary failed: {e}")

        # B.3: Correlation matrix
        st.markdown("#### Stage 2.3 Correlation Matrix")
        try:
            eda_module.show_correlation_heatmap(df)
        except Exception as e:
            st.error(f"Correlation heatmap failed: {e}")

        # Distribution of numeric features
        st.markdown("#### Stage 2.4 Distributions of Features")
        try:
            eda_module.show_distributions(df)
        except Exception as e:
            st.error(f"Distributions rendering failed: {e}")

        # Showing data summary for various Train/test split values (we haven't done actual split yet)
        st.markdown("#### Stage 2.5 Train/Test Split Summary")
        train_size = st.slider("Train size (default: 0.8)", 0.1, 0.95, st.session_state.train_size, 0.05)
        st.session_state.train_size = train_size
        try:
            eda_module.show_train_test_split_summary(df, train_size)
        except Exception as e:
            st.error(f"Train/test split summary failed: {e}")

        # Navigate to the next page(issues and preprocessing)
        st.markdown("---")
        st.markdown("### Ready to move forward?")

        # Only shows next button if data is uploaded and target column is selected
        if st.session_state.dataframe is not None and st.session_state.target_column is not None:
            if st.button("Next: Issues Detection and Preprocessing",type="primary", use_container_width=True):
                st.switch_page("pages/issues_and_preprocessing.py")
    else:
        st.info("Please upload a dataset to proceed further")


if __name__ == "__main__":
    main()
