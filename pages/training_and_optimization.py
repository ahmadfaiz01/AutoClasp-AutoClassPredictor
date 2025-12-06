import streamlit as st
import pandas as pd

from autoclasp.train import train_and_optimize_models

def show_training_and_optimization_page():
    st.title("Model Training & Hyperparameter Optimization")

    # 1. Ensuring we have train/test splits from preprocessing
    required = ["X_train", "y_train", "X_test", "y_test"]
    for split in required:
        if split not in st.session_state:
            st.session_state[split] = None
            st.warning(
                "Missing data for training: "
                + ", ".join(split)
                + ". Please complete the issues & preprocessing step first."
            )
            return      # Exit the page if data is missing

    # Passing the train data from session state
    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    # Hyperparameter optimization controls (GridSearchCV / RandomizedSearchCV)
    st.subheader("Hyperparameter Optimization Settings")

    search_type = st.radio(
        "Search strategy:",
        options=["grid", "random"],
        horizontal=True,
        help="Use Grid Search to exhaustively try a fixed grid or Randomized Search to sample from distributions.",
    )

    # Stratified k-fold cross-validation split settings (default is 5)
    cv_splits = st.slider(
        "Number of CV folds",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Stratified k-fold cross-validation.",
    )

    # Number of iterations for Randomized Search
    iterations = st.slider(
        "Randomized Search iterations",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Used only when search strategy is 'random'.",
    )

    # Primary metric for optimization during CV
    primary_metric_for_cv = st.selectbox(
        "Primary metric to optimize during CV:",
        options=["accuracy", "f1_weighted"],
        index=1,
        help="This metric is used by GridSearchCV / RandomizedSearchCV to pick the best hyperparameters.",
    )

    # Random state configuration for reproducibility
    random_state = st.number_input(
        "Random state (for reproducibility)",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    # 3. Run training and hyperparameter optimization
    if st.button("Run training and hyperparameter optimization", type="primary"):
        with st.spinner("Training and optimizing all models... Hold on tight!"):
            models_trained, results_df = train_and_optimize_models(
                X_train=X_train,
                y_train=y_train,
                search_type=search_type,
                primary_metric=primary_metric_for_cv,
                cv_splits=cv_splits,
                iterations_random_search=iterations,
                random_state=random_state,
            )

        # Saving reults in session_state for comparison, explainability, report later on
        st.session_state["trained_models"] = models_trained
        st.session_state["results_df"] = results_df

        st.success("Training & hyperparameter optimization complete.")

    # Showing Cross Validation results
    # We will have the best hyperparameters per model now
    if "results_df" in st.session_state:
        st.subheader("Cross-Validation Result (Best per Model)")
        cv_results_df: pd.DataFrame = st.session_state["results_df"].copy()

        if "best_score" in cv_results_df.columns:
            cv_results_df["best_score"] = cv_results_df["best_score"].astype(float).round(4)
        if "training_time" in cv_results_df.columns:
            cv_results_df["training_time"] = cv_results_df["training_time"].astype(float).round(3)

        with st.expander("Show CV results table", expanded=True):
            st.dataframe(cv_results_df, use_container_width=True)

        # Navigation button to Model Comparison Dashboard
        st.markdown("---")
        st.info("Training complete! Proceed to evaluate models on test data and compare their performance.")
        if st.button("Go to Model Comparison & Evaluation", type="primary", use_container_width=True):
            st.switch_page("pages/comparison_and_explainability.py")


if __name__ == "__main__":
    show_training_and_optimization_page()
