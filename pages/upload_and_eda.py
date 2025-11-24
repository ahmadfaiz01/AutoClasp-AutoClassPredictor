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
        "### Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help= "CSV, XLSX, XLS supported formats only"
    )

    # If a file is uploaded,check the type and than read it into a DataFrame
    if file_uploaded is not None:
        try:
            if file_uploaded.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_uploaded)
            else:
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
        # List of dictionary (rows:count and columns:count
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

        # Asking for the target value column
        target_col = st.selectbox(
            "Select the target column for class distribution",
            options=df.columns.tolist(),
            placeholder="Select target column"
        )
        st.session_state.target_column = target_col     # Updating target column in session state
        # Calling the show class distribution function
        eda_module.show_class_distribution(df, target_col)


    # Stage 2 - EDA
        st.title("Stage 2: Exploratory Data Analysis (EDA)")

        # Calling EDA utilities from eda_module (from eda.py)

        # Missing value analysis
        st.markdown("#### Stage 2.1: Missing Value Analysis")
        eda_module.show_missing_values(df)

        # B.2: Outlier detection (placeholder summary for now)
        st.markdown("#### Stage 2.2 Outlier Detection Summary")
        eda_module.show_outlier_summary(df)

        # B.3: Correlation matrix
        st.markdown("#### Stage 2.3 Correlation Matrix")
        eda_module.show_correlation_heatmap(df)

        # Distribution of numeric features
        st.markdown("#### Stage 2.4 Distributions of Features")
        eda_module.show_distributions(df)

        # Showing data summary for various Train/test split values (we haven't done actual split yet)
        st.markdown("#### Stage 2.5 Train/Test Split Summary")
        train_size = st.slider("Train size (default: 0.8)", 0.1, 0.9, st.session_state.train_size, 0.05)
        st.session_state.train_size = train_size
        eda_module.show_train_test_split_summary(df, train_size)

    else:
        st.info("Please upload a dataset to proceed further")

if __name__ == "__main__":
    main()