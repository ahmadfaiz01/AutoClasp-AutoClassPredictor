import time
import numpy as np
import pandas as pd

# For generating distributions and random values
from scipy.stats import randint, uniform

# For finding best hyperparameters for models
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
)

# Model algorithm to be used
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

SearchType = Literal["grid", "random"]


# Function that returns a dictionary of base models
# Structure: {model_name: model_instance}
def get_base_models(random_state: int):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM": SVC(
            probability=True,
            random_state=random_state,
        ),
        "Rule-based (Stratified Baseline)": DummyClassifier(
            strategy="stratified",
            random_state=random_state,
        ),
    }


# Function that returns a dictionary of parameter grids for grid search
# Structure: {model_name: {param_name: [param_values]}}
def get_param_grids():
    return {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Naive Bayes": {
            "var_smoothing": np.logspace(-9, -7, 3),
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "SVM": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "Rule-based": {}                            # No hyperparameters
    }


# Function that returns a dictionary of parameter distributions for random search
# Structure: {model_name: {param_name: distribution values}}
def get_param_distributions():
    return {
        # Hyperparameter for Logistic Regression
        "Logistic Regression": {
            "C": uniform(0.001, 100.0),             # from 0.001 to 100.001
        },
        "K-Nearest Neighbors": {
            "n_neighbors": randint(3, 30),          # random integer from 3 to 29
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "max_depth": [None, 3, 5, 10, 20],      # including None for unlimited depth
            "min_samples_split": randint(2, 20),    # random integer from 2 to 19
            "min_samples_leaf": randint(1, 10),     # random integer from 1 to 9
        },
        "Naive Bayes": {
            "var_smoothing": uniform(1e-10, 1e-6),  # from 1e-10 to 1.0001e-6
        },
        "Random Forest": {
            "n_estimators": randint(50, 300),       # random integer from 50 to 299
            "max_depth": [None, 5, 10, 20],         # including None for unlimited depth
            "min_samples_split": randint(2, 10),    # random integer from 2 to 9
            "min_samples_leaf": randint(1, 10),     # random integer from 1 to 9
        },
        "SVM": {
            "C": uniform(0.01, 100.0),              # from 0.01 to 100.01
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "Rule-based": {}                            # No hyperparameters
    }


def train_and_optimize_models(
    X_train,
    y_train,
    search_type: SearchType = "grid",
    primary_metric: str = "f1_weighted",
    cv_splits: int = 5,
    n_iter_random: int = 20,
    random_state: int = 42 ):

    models = get_base_models(random_state)
    grids = get_param_grids()
    dists = get_param_distributions()

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Dictionary storing best models and list of its results
    best_models = {}
    # List to store cross-validation results
    records = []

    for name, base_model in models.items():
        # Current model under training
        print(f"Training model: {name}")

        # Fetching parameters for the current model
        param_grid = grids[name]
        param_dist = dists[name]

        # If no hyperparameters are there, it would be a rule based model
        # So we train it without any hyperparameter optimization
        if not param_grid and not param_dist:
            # Record Starting time
            start = time.time()

            # Fitting the model on training data
            base_model.fit(X_train, y_train)

            # Calculating training time taken
            training_time = time.time() - start
            setattr(base_model, "_training_time", training_time)

            # Storing the model's name and results in our dictionary
            best_models[name] = base_model
            records.append(
                {
                    "model": name,
                    "best_score": np.nan,
                    "best_params": {},
                    "training_time": training_time,
                    "search_type": "none",
                }
            )
            continue

        # If search type is grid, we use GridSearchCV
        if search_type == "grid":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=primary_metric,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
        # If search type is random,we use RandomizedSearchCV
        else:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist if param_dist else param_grid,
                n_iter=n_iter_random,
                scoring=primary_metric,
                cv=cv,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
            )

        # Fitting the search on training data and recording time taken
        start = time.time()
        search.fit(X_train, y_train)
        training_time = time.time() - start

        # Fetching the best estimator from the search
        best_estimator = search.best_estimator_
        setattr(best_estimator, "_training_time", training_time)

        # Storing the model's name and results in our dictionary
        best_models[name] = best_estimator
        records.append(
            {
                "model": name,
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "training_time": training_time,
                "search_type": search_type,
            }
        )

    # Creating a Record DataFrame from the records
    results_df = pd.DataFrame(records)
    return best_models, results_df