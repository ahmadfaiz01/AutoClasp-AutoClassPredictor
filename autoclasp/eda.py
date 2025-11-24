from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    # List of dictionaries with keys (column and outlier count)
    outlier_data = []

    # Selecting numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Removing NaN values and storing in temp df
    cleaned_df = df.dropna()

    # Calculating IQR for each column
    for col in numeric_df.columns:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1

        # If a column has no IQR, set its outlier count to 0
        if iqr == 0:
            # Saving the outlier info into list
            outlier_data.append({"column": col, "outlier_count": 0})
            continue

        # Defining outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Flagging outliers
        # For a particular column, go into column we are in and mark datapoints as outliers which out of bounds
        outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]

        # Storing into the list
        outlier_data.append({"column": col, "outlier_count": int(outliers.sum())})

    # Returning the list as a dataframe so streamlit interpret it
    return pd.DataFrame(outlier_data)
def show_outlier_summary(df: pd.DataFrame) -> None:

    # Calling the outlier detection functions
    iqr_df = detect_outliers_iqr(df)
    z_df = detect_outliers_zscore(df)

    # Displaying the outlier summaries
    st.markdown("**IQR-based outliers (per numeric column)**")
    st.dataframe(iqr_df)

    st.markdown("**Z-score-based outliers (per numeric column)**")
    st.dataframe(z_df)




# Function to show correlation matrix as a heatmap
def show_correlation_heatmap(df: pd.DataFrame) -> None:
    # Selecting only numeric columns in df
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        st.info("No numeric columns available for correlation heatmap.")
    else:
        # Computing the correlation matrix
        correlation_matrix = numeric_df.corr()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            center=0,
            annot=True,
            ax=ax
        )

        # Setting title and layout
        ax.set_title("Correlation Matrix")
        fig.tight_layout()

        # Displaying the heatmap on app
        st.pyplot(fig)



# Funtion to show class distribution in target column
def show_class_distribution(df: pd.DataFrame, target_col: str) -> None:
    # Validating if target_col exists in df
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the dataframe.")
    else:
        class_counts = df[target_col].value_counts(dropna=False)    #Didn't drop empty tuples to show real data distribution
        class_percentages = df[target_col].value_counts(normalize=True, dropna=False) * 100  #Normalizing so all%ages sum to 100

        distribution_summary = pd.DataFrame(
            {
                "class": class_counts.index.astype(str),
                "count": class_counts,
                "percentage": class_percentages,
            }
        )

        # Displaying the class distribution analytics
        st.write("### Class Distribution Table:")
        st.dataframe(distribution_summary)

        # Plotting the class distribution
        fig = px.bar(
            distribution_summary,
            x="class",
            y="count",
            text=distribution_summary["percentage"].round(2).astype(str) + "%",
            title=f"Class distribution: {target_col}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="Class", yaxis_title="Count")

        # Rendering it in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)



# Function to show missing values analysis
def show_missing_values(df: pd.DataFrame) -> None:
    # Analyzes missing values per column
    total_missing = df.isnull().sum()
    percent_missing = (total_missing / len(df)) * 100

    # Total data tuples in df
    total_data = df.shape[0] * df.shape[1]

    # Dataframe for missing value analytics
    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": total_missing.values,
            "missing_percentage": percent_missing.values,
        }
    )

    # Global missing data percentage
    golbal_missing_percentage = total_missing.values.sum() / total_data * 100

    # Displaying the missing value summary
    st.metric(
        label="Overall Missing Data Percentage",
        value=f"{golbal_missing_percentage:.2f}%",
    )
    st.dataframe(missing_summary)



# Function to show train-test data size summary
def show_train_test_split_summary(df: pd.DataFrame, train_size: float) -> None:
    # Total rows
    total_rows = df.shape[0]
    # Training rows
    train_rows = int(total_rows * train_size)
    # Testing rows
    test_rows = total_rows - train_rows

    # Displaying the train-test split summary
    st.write(f"Train Size: {train_size*100:.2f}% \t Train Rows: {train_rows}")
    st.write(f"Test Size: {(1-train_size)*100:.2f}% \t Test Rows: {test_rows}")