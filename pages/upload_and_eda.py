import streamlit as st
import pandas as pd

# Importing EDA related utilities
from autoclasp import eda as eda_module

def main():
    # Stage 1 - Dataset Uploading and Basic Information
    st.title("Stage 1: Dataset Uploading and Basic Information")

    # Asking User to upload a file
    st.markdown("### Stage 1.1: Upload Your Dataset")

    # Handling the input of the dataset
    file_uploaded = st.file_uploader(
        "Upload your dataset (CSV, XLSX, XLS supported)",
        type=["csv", "xlsx", "xls"],
        help= "CSV, XLSX, XLS supported formats only"
    )

    # If a file is uploaded, read it into a DataFrame
    if file_uploaded is not None:
        try:
            df = pd.read_csv(file_uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    # Storing the uploaded file and dataframe in session state
        st.session_state.uploaded_file = file_uploaded
        st.session_state.dataframe = df

        st.success("File uploaded successfully!")

    # A2 - Basic Data Information
        st.markdown("### Stage 1.2: Basic Data Information")
        # No need to except handle since it is already done above
        # Code would not reach here of file is not uploaded and df is not created

        st.write("Here is a preview of your dataset:")
        st.dataframe(df.head(5))  # Displaying first 5 rows of the dataframe

        # No. of Rows and Columns
        st.write(f"Rows : {df.shape[0]}")
        st.write(f"Columns : {df.shape[1]}")

        # Feature Names and Datatypes
        st.write("### Feature Names and Data Types")
        st.write("\n Column \t\t\t Data Type")
        for col in df.columns:
            st.write(f"- {col} \t\t\t {df[col].dtype}\n")

        # Data Statistics
        st.write("### Data Statistics")
        st.write(df.describe(include='all').T)  # caters to both numerical and categorical data

        # Checking Class Distribution (whether balanced or not)
        st.write("### Class Distribution")

        # Asking for the target value column
        target_col = st.selectbox(
            "Select the target column for class distribution",
            options=df.columns.tolist(),
            placeholder="Select target column"
        )
        st.session_state.target_column = target_col

        if target_col is not None:
            eda_module.plot_class_distribution(df, target_col)
        else:
            st.info("Please select a target column to view class distribution.")

        st.markdown("-End of Stage 1-")

    # Stage 2 - EDA
        st.markdown("###Stage2: Exploratory Data Analysis (EDA)")

        # Calling EDA utilities from eda_module (from eda.py)

        # Missing value analysis
        st.markdown("#### Missing Value Analysis")
        eda_module.show_missing_values(df)

        # B.2: Outlier detection (placeholder summary for now)
        st.markdown("#### Outlier Detection Summary")
        eda_module.show_outlier_summary(df)

        # B.3: Correlation matrix
        st.markdown("#### Correlation Matrix")
        eda_module.show_correlation_matrix(df)

        # Distribution of numeric features
        st.markdown("#### Distributions of Features")
        eda_module.show_distributions(df)

        # Train/test split summary (we haven't done actual split yet)
        st.markdown("#### Train/Test Split Summary")
        train_size = st.session_state.train_size
        eda_module.show_train_test_split_summary(df, train_size=train_size)

    else:
        st.info("Please upload a dataset to proceed further")

if __name__ == "__main__":
    main()