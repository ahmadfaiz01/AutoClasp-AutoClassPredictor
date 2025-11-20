# AutoClasp \- AutoML for Classification

**Members:**
- Muhammad Ahmad
- Arham Ali

AutoClasPredictor a Streamlit web app that does AutoML for **classification**.  
Idea is: you upload a CSV, the app does EDA, checks data quality, lets you pick preprocessing options, trains multiple models, compares them, and then gives you a final report.

## What this app is supposed to do

\- Upload a CSV and pick the target column  
\- Show basic info (rows, columns, dtypes, summary stats, class distribution)  
\- Run automatic EDA:
  \- missing values  
  \- outliers (IQR + Z\-score)  
  \- correlations  
  \- distributions for numeric and categorical features  
\- Detect issues:
  \- missing values / outliers / imbalance  
  \- high\-cardinality categoricals  
  \- constant features / duplicates  
  \- compute a Data Health Score (0\-100)  
\- Let the user choose preprocessing:
  \- imputation, scaling, encoding, outlier handling, train/test split  
\- Train several classifiers:
  \- Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, SVM, rule\-based tree  
  \- with GridSearchCV / RandomizedSearchCV  
\- Show metrics:
  \- accuracy, precision, recall, F1, ROC\-AUC, confusion matrix, training time  
  \- comparison table + plots + ROC curves  
\- Extra stuff:
  \- feature importance + permutation importance  
  \- ensemble models (voting, maybe stacking)  
  \- model stability across different random seeds  
  \- auto\-generated report (Markdown/HTML, maybe PDF later)

Right now I\'m building this step by step in Streamlit with a modular Python structure.

## How to run it locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
