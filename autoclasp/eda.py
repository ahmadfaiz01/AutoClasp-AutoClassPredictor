from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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


def show_train_test_split_summary(df: pd.DataFrame, train_size: float) -> None:
    total_rows = df.shape[0]
    train_rows = int(total_rows * train_size)
    test_rows = total_rows - train_rows

    st.write(f"Train Size: {train_size*100:.2f}% \t Train Rows: {train_rows}")
    st.write(f"Test Size: {(1-train_size)*100:.2f}% \t Test Rows: {test_rows}")