# Importing necessary libraries
import streamlit as st

# Setting up the Streamlit app

def init_page():
    st.set_page_config(
        page_title="AutoCLASP - Auto Class Predictor",
        page_icon="data/logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Safe logo loading
    try:
        st.logo("data/logo.png")
    except Exception:
        st.sidebar.image("data/logo.png", width=180)

def init_session_state():
    # Ensuring keys exist in session state, if they dont set default values
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    if "train_size" not in st.session_state:
        st.session_state.train_size = 0.8


def pipeline_timeline():
    st.sidebar.title("Pipeline Timeline")
    steps = [
        "1. Upload",
        "2. EDA",
        "3. Issues Detection",
        "4. PreProcessing",
        "5. Training"
        "6. Evaluation",
        "7. Best Model Selection",
        "8. Report"
    ]
    for step in steps:
        st.sidebar.markdown(f"- {step}")

def main():

    # Calling our initialization functions
    init_page()
    init_session_state()
    pipeline_timeline()

    st.title("Lets get started with AutoCLASP!")

    st.markdown(
        "Use the pages in the sidebar to:\n"
        "1\. Upload your dataset and run automated EDA.\n"
        "2\. Detect issues and configure preprocessing.\n"
        "3\. Train, optimize, and compare models.\n"
        "4\. View explanations and download a final report."
    )
    st.info("Start with the `Upload \& EDA` page from the left sidebar.")

if __name__ == "__main__":
    main()