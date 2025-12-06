import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from autoclasp.evaluate import evaluate_all_models
from typing import Dict, Tuple, List

# Color scheme for all charts - keeping it simple and consistent
COLORS = {
    'blue': '#87CEEB',      # sky blue for accuracy
    'green': '#50C878',     # for precision and positive indicators
    'red': '#E74C3C',       # for f1 score
    'orange': '#FF8C42',    # for training time
    'grey': '#95A5A6'       # for recall and secondary metrics
}


def create_metrics_comparison_chart(results_df, metric, title):
    # Sort models by their performance on this metric
    sorted_df = results_df.sort_values(by=metric, ascending=True)
    
    # Pick color based on what we're measuring
    if metric == 'accuracy':
        bar_color = COLORS['blue']
    elif metric == 'precision_weighted':
        bar_color = COLORS['green']
    elif metric == 'recall_weighted':
        bar_color = COLORS['grey']
    elif metric == 'f1_weighted':
        bar_color = COLORS['red']
    else:
        bar_color = COLORS['blue']
    
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_df['model'],
            x=sorted_df[metric],
            orientation='h',
            marker=dict(color=bar_color, line=dict(color=bar_color, width=0)),
            text=[f"{val:.3f}" for val in sorted_df[metric]],
            textposition='outside',
            textfont=dict(color='#FAFAFA', size=11),
            hovertemplate='<b>%{y}</b><br>' + metric + ': %{x:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#FAFAFA'), x=0.5, xanchor='center'),
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title='',
        height=380,
        showlegend=False,
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        margin=dict(l=140, r=40, t=70, b=50),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)', showgrid=True),
        yaxis=dict(showgrid=False)
    )
    
    return fig


def create_overall_comparison_chart(results_df):
    # Using the same 4 colors consistently across all charts
    metrics = [
        ('accuracy', 'Accuracy', COLORS['blue']),
        ('f1_weighted', 'F1 Score', COLORS['red']),
        ('precision_weighted', 'Precision', COLORS['green']),
        ('recall_weighted', 'Recall', COLORS['grey'])
    ]
    
    fig = go.Figure()
    
    # Add a bar for each metric
    for metric, label, color in metrics:
        fig.add_trace(go.Bar(
            name=label,
            x=results_df['model'],
            y=results_df[metric],
            marker=dict(color=color),
            text=[f"{val:.2f}" for val in results_df[metric]],
            textposition='outside',
            textfont=dict(size=9, color='#FAFAFA'),
            hovertemplate=f'<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Performance Overview', font=dict(size=20, color='#FAFAFA'), x=0.5, xanchor='center'),
        xaxis_title='',
        yaxis_title='Score',
        barmode='group',
        height=480,
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)', range=[0, 1.1], showgrid=True),
        margin=dict(l=50, r=50, t=90, b=50)
    )
    
    return fig


def create_roc_comparison_chart(results_df):
    # Some models might not have ROC-AUC (like in multi-class), so handle that
    display_df = results_df.copy()
    display_df['roc_auc_display'] = display_df['roc_auc'].fillna(0)
    sorted_df = display_df.sort_values(by='roc_auc_display', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_df['model'],
            x=sorted_df['roc_auc_display'],
            orientation='h',
            marker=dict(color=COLORS['green'], line=dict(color=COLORS['green'], width=0)),
            text=[f"{val:.3f}" if not pd.isna(val) else "N/A" for val in sorted_df['roc_auc']],
            textposition='outside',
            textfont=dict(color='#FAFAFA', size=11),
            hovertemplate='<b>%{y}</b><br>ROC-AUC: %{x:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text='ROC-AUC Comparison', font=dict(size=18, color='#FAFAFA'), x=0.5, xanchor='center'),
        xaxis_title='ROC-AUC Score',
        yaxis_title='',
        height=380,
        showlegend=False,
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        margin=dict(l=140, r=40, t=70, b=50),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)', range=[0, 1.05], showgrid=True),
        yaxis=dict(showgrid=False)
    )
    
    return fig


def create_training_time_chart(results_df):
    # Showing which models are fast vs slow to train
    sorted_df = results_df.sort_values(by='training_time', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_df['model'],
            x=sorted_df['training_time'],
            orientation='h',
            marker=dict(color=COLORS['orange'], line=dict(color=COLORS['orange'], width=0)),
            text=[f"{val:.2f}s" for val in sorted_df['training_time']],
            textposition='outside',
            textfont=dict(color='#FAFAFA', size=11),
            hovertemplate='<b>%{y}</b><br>Training Time: %{x:.2f}s<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text='Training Time Comparison', font=dict(size=18, color='#FAFAFA'), x=0.5, xanchor='center'),
        xaxis_title='Time (seconds)',
        yaxis_title='',
        height=380,
        showlegend=False,
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        margin=dict(l=140, r=40, t=70, b=50),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)', showgrid=True),
        yaxis=dict(showgrid=False)
    )
    
    return fig


def rank_models(results_df, primary_metric='f1_weighted'):
    # Just sorting models and adding rank numbers
    ranked_df = results_df.copy()
    ranked_df = ranked_df.sort_values(by=primary_metric, ascending=False)
    ranked_df.insert(0, 'rank', range(1, len(ranked_df) + 1))
    return ranked_df


def generate_confusion_insights(confusion_matrix, model_name, class_names=None):
    # If no class names provided, just number them
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
    
    # Basic accuracy from confusion matrix
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    accuracy = correct / total if total > 0 else 0
    
    n_classes = len(confusion_matrix)
    
    # Check which class the model is good at and which one it struggles with
    class_recalls = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        total_actual = confusion_matrix[i, :].sum()
        recall = tp / total_actual if total_actual > 0 else 0
        class_recalls.append((class_names[i], recall))
    
    class_recalls.sort(key=lambda x: x[1], reverse=True)
    best_class = class_recalls[0]
    worst_class = class_recalls[-1]
    
    # Find where the model makes the most mistakes
    cm_copy = confusion_matrix.copy()
    np.fill_diagonal(cm_copy, 0)
    confusion_msg = ""
    if cm_copy.max() > 0:
        max_confusion = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        actual_class = class_names[max_confusion[0]]
        predicted_class = class_names[max_confusion[1]]
        confusion_msg = f"Most confused: {actual_class} predicted as {predicted_class}"
    
    return {
        'accuracy': f"{accuracy:.1%}",
        'best_class': f"{best_class[0]} performs best ({best_class[1]:.1%} recall)",
        'worst_class': f"{worst_class[0]} needs improvement ({worst_class[1]:.1%} recall)",
        'confusion': confusion_msg
    }


def show_comparison_page():
    st.title("Model Comparison Dashboard")
    st.markdown("Compare trained models and analyze their performance")
    
    # Custom CSS to make metrics look better with our color scheme
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 26px;
            color: #87CEEB;
        }
        .stSlider > div > div > div > div {
            background-color: #87CEEB;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Make sure we have trained models before showing anything
    if 'trained_models' not in st.session_state:
        st.warning("No trained models found. Please train models first.")
        if st.button("Go to Training Page"):
            st.switch_page("pages/training_and_optimization.py")
        return
    
    # Need test data to evaluate models
    if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
        st.error("Test data not found. Please ensure data is properly split.")
        return
    
    # Get everything from session
    trained_models = st.session_state.trained_models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    st.success(f"Found {len(trained_models)} trained models")
    
    # Big button to start evaluation
    if st.button("Evaluate All Models", type="primary", use_container_width=True):
        with st.spinner("Evaluating models..."):
            results_df, confusion_figs = evaluate_all_models(trained_models, X_test, y_test)
            st.session_state.evaluation_metrics = results_df
            st.session_state.confusion_matrices = confusion_figs
            st.success("Evaluation complete")
    
    # Don't show anything until evaluation is done
    if 'evaluation_metrics' not in st.session_state:
        st.info("Click the button above to evaluate all models")
        return
    
    results_df = st.session_state.evaluation_metrics
    confusion_figs = st.session_state.confusion_matrices
    
    st.markdown("---")
    
    # Let user pick what metric matters most to them
    st.subheader("Select Primary Metric")
    metric_options = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc']
    primary_metric = st.selectbox("Ranking metric:", options=metric_options, index=1)
    
    st.markdown("---")
    st.subheader("Model Rankings")
    
    # Rank and format the results table
    ranked_df = rank_models(results_df, primary_metric)
    display_df = ranked_df.copy()
    
    # Make numbers look cleaner
    numeric_cols = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc', 'training_time', 'inference_time']
    for col in numeric_cols:
        if col in display_df.columns:
            if col in ['training_time', 'inference_time']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}s")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, height=320)
    
    st.markdown("---")
    st.subheader("Export Results")
    
    # Download buttons for CSV and Excel
    col1, col2 = st.columns(2)
    
    with col1:
        csv = ranked_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        try:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                ranked_df.to_excel(writer, index=False, sheet_name='Comparison')
            
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name="model_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("Install openpyxl for Excel export")
    
    st.markdown("---")
    st.subheader("Performance Overview")
    # Main chart showing all 4 key metrics together
    st.plotly_chart(create_overall_comparison_chart(results_df), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Detailed Analysis")
    
    # Tabs for diving deeper into individual metrics
    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "F1 Score", "ROC-AUC", "Training Time"])
    
    with tab1:
        st.plotly_chart(create_metrics_comparison_chart(results_df, 'accuracy', 'Accuracy Comparison'), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_metrics_comparison_chart(results_df, 'f1_weighted', 'F1 Score Comparison'), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_roc_comparison_chart(results_df), use_container_width=True)
    
    with tab4:
        st.plotly_chart(create_training_time_chart(results_df), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Confusion Matrix Analysis")
    
    # Show confusion matrix and insights for each model
    for model_name, fig in confusion_figs.items():
        with st.expander(f"{model_name}", expanded=False):
            st.plotly_chart(fig, use_container_width=True)
            
            # Extract the actual matrix data from the plotly figure
            cm_data = fig.data[0].z
            insights = generate_confusion_insights(np.array(cm_data), model_name)
            
            # Show insights in a clean table
            insight_table = pd.DataFrame([
                {"Metric": "Overall Accuracy", "Value": insights['accuracy']},
                {"Metric": "Best Performance", "Value": insights['best_class']},
                {"Metric": "Needs Attention", "Value": insights['worst_class']},
                {"Metric": "Common Error", "Value": insights['confusion']}
            ])
            
            st.dataframe(insight_table, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Summary")
    
    # Quick metrics showing best models
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        st.metric("Best Accuracy", f"{best_accuracy['accuracy']:.3f}", best_accuracy['model'])
    
    with col2:
        best_f1 = results_df.loc[results_df['f1_weighted'].idxmax()]
        st.metric("Best F1 Score", f"{best_f1['f1_weighted']:.3f}", best_f1['model'])
    
    with col3:
        if not results_df['roc_auc'].isna().all():
            best_roc = results_df.loc[results_df['roc_auc'].idxmax()]
            st.metric("Best ROC-AUC", f"{best_roc['roc_auc']:.3f}", best_roc['model'])
        else:
            st.metric("Best ROC-AUC", "N/A", "Multi-class")
    
    # Show top 3 based on selected metric
    st.markdown("**Top 3 Models**")
    top_3 = ranked_df.head(3)
    for idx, row in top_3.iterrows():
        st.write(f"{row['rank']}. {row['model']} - {primary_metric}: {row[primary_metric]:.4f}")
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Back to Training", use_container_width=True):
            st.switch_page("pages/training_and_optimization.py")
    
    with col2:
        if st.button("Generate Report", use_container_width=True, type="primary"):
            st.switch_page("pages/final_report.py")


if __name__ == "__main__":
    show_comparison_page()