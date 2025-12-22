import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Function to detect outliers using IQR method
def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    # List of dictionaries with keys (column, outlier count, percentage)
    outlier_data = []

    # Selecting numeric columns
    numeric_df = df.select_dtypes(include=[np.number])


    # Calculating IQR for each column
    for col in numeric_df.columns:
        # Removing NaN values of the column and storing in temp series
        column_values = numeric_df[col].dropna()

        # Gives us the value below which 25% of data lies
        q1 = column_values.quantile(0.25)

        # Gives us the value below which 75% of data lies
        q3 = column_values.quantile(0.75)
        iqr = q3 - q1

        # If a column has no IQR, set its outlier count to 0
        if iqr == 0:
            # Saving the outlier info into list
            outlier_data.append({"column": col, "outlier_count": 0, "percentage":0})
            continue

        # Defining outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Flagging outliers
        # For a particular column, go into column we are in and mark datapoints as outliers which are out of bounds
        outliers = (column_values < lower_bound) | (column_values > upper_bound)

        # Storing into the list
        outlier_data.append({"column": col, "outlier_count": int(outliers.sum()), "percentage":((outliers.sum()) / len(column_values)) * 100})

    # Returning the list as a dataframe so streamlit interpret it
    return pd.DataFrame(outlier_data)

# Function to detect outliers using Z-Score method
def detect_outliers_zscore(df: pd.DataFrame) -> pd.DataFrame:
    # List of dictionaries with keys (column, outlier count, percentage)
    outlier_data = []

    # Selecting numeric columns
    numeric_df = df.select_dtypes(include=[np.number])


    # Iterating over columns and finding their outlier counts with Z-Score
    for col in numeric_df.columns:
        # Removing NaN values from column and storing in temp series
        column_values = numeric_df[col].dropna()

        # Calculating mean for each column
        mean = column_values.mean()
        std = column_values.std()

        # If std is 0, set outlier count to 0
        if std == 0:
            outlier_data.append(({"column": col, "outlier_count": 0, "percentage":0}))
        else:
            # Calculating Z-Scores
            z = (column_values - mean) / std

            # Marking values as outliers with Absolute Z-Score > 3
            outliers = z.abs() > 3
            # Storing columns info into the list
            outlier_data.append({"column": col, "outlier_count": int(outliers.sum()), "percentage":((outliers.sum()) / len(column_values)) * 100})

    return pd.DataFrame(outlier_data)

# Function to show outlier summary using both IQR and Z-Score methods
def show_outlier_summary(df: pd.DataFrame) -> None:

    # Calling the outlier detection functions
    iqr_df = detect_outliers_iqr(df)
    z_df = detect_outliers_zscore(df)

    # Displaying the outlier summaries
    st.markdown("**IQR-based outliers (per numeric column)**")
    st.dataframe(iqr_df)

    st.markdown("**Z-score-based outliers (per numeric column)**")
    st.dataframe(z_df)


def show_distributions(df: pd.DataFrame) -> None:
# Histogram for Numeric Values
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.info("No numeric columns available for distribution plots.")
    else:
        for col in numeric_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(numeric_df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

# Bar Plots for Categorical Values
    categorical_df = df.select_dtypes(include=['object', 'category', 'bool'])
    if categorical_df.empty:
        st.info("No categorical columns available for distribution plots.")
    else:
        for col in categorical_df.columns:
            # Build a stable dataframe with explicit column names
            vc = categorical_df[col].value_counts(dropna=False).reset_index(name='count')
            vc = vc.rename(columns={'index': col})
            # compute percentage for labels
            vc['percentage'] = (vc['count'] / vc['count'].sum()) * 100

            # Use explicit column names for px.bar
            fig = px.bar(
                vc,
                x=col,
                y='count',
                text=vc['percentage'].round(2).astype(str) + '%',
                title=f"Distribution of {col}",
            )
            fig.update_layout(xaxis_title=col, yaxis_title="Count")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

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


# This is function for Task A
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
            title=f"Class distribution of {target_col}",
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
    global_missing_percentage = total_missing.values.sum() / total_data * 100

    # Displaying the missing value summary
    st.metric(
        label="Overall Missing Data Percentage",
        value=f"{global_missing_percentage:.2f}%",
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


# Function to process data to resolve plotting issues
def report_non_numeric(df: pd.DataFrame, sample_limit: int = 10) -> pd.DataFrame:
    """Return a small DataFrame reporting columns with non-numeric entries."""
    rows = []
    for col in df.columns:
        coerced = pd.to_numeric(df[col].astype(str).str.strip().replace(',', '.'), errors='coerce')
        n_invalid = coerced.isna().sum() - df[col].isna().sum()
        if n_invalid > 0:
            sample_vals = list(pd.Series(df[col].astype(str).unique())[:sample_limit])
            rows.append({"column": col, "non_numeric_count": int(n_invalid), "sample_values": sample_vals})
    return pd.DataFrame(rows)

def clean_and_coerce(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Clean string columns and coerce numeric-like columns to numeric.
    Columns where at least `threshold` fraction become numeric are converted.
    Returns a new DataFrame (doesn't mutate original).
    """
    df2 = df.copy()
    # Basic string cleaning
    for col in df2.select_dtypes(include=['object', 'string']).columns:
        s = df2[col].astype(str)
        s = s.str.strip()
        s = s.str.replace(r'\s+', ' ', regex=True)
        s = s.str.replace(',', '.', regex=False)  # comma decimal -> dot
        s = s.replace({'nan': pd.NA})
        df2[col] = s

    # Try to coerce columns to numeric when most values make sense
    for col in df2.columns:
        coerced = pd.to_numeric(df2[col], errors='coerce')
        valid_frac = coerced.notna().mean()
        if valid_frac >= threshold:
            df2[col] = coerced

    return df2