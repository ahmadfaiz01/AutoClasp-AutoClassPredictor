import pandas as pd
import numpy as np
import plotly.graph_objs as go
from typing import Any, Dict, Optional


def get_feature_importance(model: Any, feature_names: list) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from a trained model if available.
    
    Supports models with:
    - feature_importances_ attribute (tree-based models)
    - coef_ attribute (linear models)
    
    Args:
        model: Trained sklearn model
        feature_names: List of feature names
    
    Returns:
        DataFrame with features and their importance scores, or None if not available
    """
    importance_scores = None
    
    # Check for tree-based models (Random Forest, Decision Tree, etc.)
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
    
    # Check for linear models (Logistic Regression, SVM with linear kernel)
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        
        # Handle multi-class case (average absolute coefficients across classes)
        if coef.ndim > 1:
            importance_scores = np.mean(np.abs(coef), axis=0)
        else:
            importance_scores = np.abs(coef)
    
    # Return None if model doesn't support feature importance
    if importance_scores is None:
        return None
    
    # Create DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, top_n: int = 20) -> go.Figure:
    """
    Create a horizontal bar chart for feature importance visualization.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        model_name: Name of the model for the chart title
        top_n: Number of top features to display
    
    Returns:
        Plotly Figure object
    """
    # Select top N features
    top_features = importance_df.head(top_n).copy()
    
    # Sort by importance for horizontal bar chart (ascending for better visualization)
    top_features = top_features.sort_values(by='importance', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                y=top_features['feature'],
                x=top_features['importance'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=top_features['importance'].round(4),
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        title=f"Feature Importance - {model_name}",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(400, top_n * 25),  # Dynamic height based on number of features
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def get_model_interpretation_text(model: Any, model_name: str) -> str:
    """
    Generate human-readable interpretation text for different model types.
    
    Args:
        model: Trained sklearn model
        model_name: Name of the model
    
    Returns:
        String with interpretation guidance
    """
    interpretations = {
        "Logistic Regression": (
            "Logistic Regression is a linear model that assigns weights (coefficients) to each feature. "
            "Features with larger absolute coefficient values have more influence on predictions. "
            "Positive coefficients increase the probability of the positive class, while negative "
            "coefficients decrease it."
        ),
        "Decision Tree": (
            "Decision Trees make predictions by asking a series of yes/no questions about features. "
            "Feature importance is calculated based on how much each feature decreases impurity "
            "(e.g., Gini impurity) when used for splitting. Features that appear higher in the tree "
            "and split large groups of samples are more important."
        ),
        "Random Forest": (
            "Random Forest builds multiple decision trees and averages their predictions. "
            "Feature importance is the average importance across all trees in the forest. "
            "Features that consistently help trees make better splits are ranked higher."
        ),
        "K-Nearest Neighbors": (
            "K-Nearest Neighbors makes predictions based on the K most similar samples in the training set. "
            "This model doesn't provide built-in feature importance scores, as it treats all features "
            "equally based on distance calculations. However, feature scaling can significantly impact "
            "which features influence predictions more."
        ),
        "SVM": (
            "Support Vector Machines find a hyperplane that best separates classes. For linear kernels, "
            "the coefficients indicate feature importance - features with larger absolute coefficients "
            "contribute more to the decision boundary. For non-linear kernels (e.g., RBF), feature "
            "importance is not directly interpretable."
        ),
        "Naive Bayes": (
            "Naive Bayes calculates the probability of each class given the features, assuming features "
            "are independent. This model doesn't provide traditional feature importance scores. Each "
            "feature contributes to predictions through its conditional probability distribution for "
            "each class."
        ),
        "Rule-based (Stratified Baseline)": (
            "The rule-based baseline classifier makes predictions by randomly sampling from the training "
            "class distribution. It doesn't use features for predictions, so feature importance is not "
            "applicable. This model serves as a simple benchmark."
        )
    }
    
    # Return interpretation if available, otherwise generic message
    return interpretations.get(
        model_name,
        "This model makes predictions based on patterns learned from the training data. "
        "Feature importance may not be directly available for this model type."
    )


def extract_model_explainability(models: Dict[str, Any], feature_names: list) -> Dict[str, Dict]:
    """
    Extract explainability information for all models.
    
    Args:
        models: Dictionary mapping model names to trained model objects
        feature_names: List of feature names from the training data
    
    Returns:
        Dictionary containing explainability information for each model:
        {
            model_name: {
                'importance_df': DataFrame with feature importance,
                'interpretation': String with model interpretation,
                'supports_importance': Boolean indicating if importance is available
            }
        }
    """
    explainability_data = {}
    
    for model_name, model in models.items():
        # Get feature importance
        importance_df = get_feature_importance(model, feature_names)
        
        # Get interpretation text
        interpretation = get_model_interpretation_text(model, model_name)
        
        # Store results
        explainability_data[model_name] = {
            'importance_df': importance_df,
            'interpretation': interpretation,
            'supports_importance': importance_df is not None
        }
    
    return explainability_data