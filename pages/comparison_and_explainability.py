import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from autoclasp.evaluate import evaluate_all_models

# Color scheme (kept consistent)
COLORS = {
    'blue': '#87CEEB',
    'green': '#50C878',
    'red': '#E74C3C',
    'orange': '#FF8C42',
    'grey': '#95A5A6'
}

# Reusable charts (kept focused and non-redundant)
def create_overall_comparison_chart(results_df):
    metrics = [
        ('accuracy', 'Accuracy', COLORS['blue']),
        ('f1_weighted', 'F1 Score', COLORS['red']),
        ('precision_weighted', 'Precision', COLORS['green']),
        ('recall_weighted', 'Recall', COLORS['grey'])
    ]
    fig = go.Figure()
    for metric, label, color in metrics:
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=results_df['model'],
                y=results_df[metric].fillna(0),
                marker=dict(color=color),
                text=[f"{val:.2f}" if pd.notna(val) else "N/A" for val in results_df[metric]],
                textposition='outside',
                textfont=dict(size=9, color='#FAFAFA'),
                hovertemplate=f'<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>'
            ))
    fig.update_layout(
        title=dict(text='Performance Overview (Grouped)', font=dict(size=18, color='#FAFAFA'), x=0.5),
        barmode='group',
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        height=460,
        margin=dict(l=40, r=40, t=80, b=120),
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(range=[0, 1.05], gridcolor='rgba(255,255,255,0.06)')
    )
    return fig

def create_training_time_chart(results_df):
    if 'training_time' not in results_df.columns:
        return None
    sorted_df = results_df.sort_values(by='training_time', ascending=True)
    fig = go.Figure(go.Bar(
        y=sorted_df['model'],
        x=sorted_df['training_time'],
        orientation='h',
        marker=dict(color=COLORS['orange']),
        text=[f"{val:.2f}s" for val in sorted_df['training_time']],
        textposition='outside',
        textfont=dict(size=11, color='#FAFAFA'),
        hovertemplate='<b>%{y}</b><br>Training Time: %{x:.2f}s<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Training Time (seconds)', font=dict(size=16, color='#FAFAFA'), x=0.5),
        template='plotly_dark',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        height=380,
        margin=dict(l=140, r=40, t=70, b=40)
    )
    return fig

def rank_models(results_df, primary_metric='f1_weighted'):
    if primary_metric not in results_df.columns:
        primary_metric = results_df.columns[1] if len(results_df.columns) > 1 else results_df.columns[0]
    ranked_df = results_df.copy().sort_values(by=primary_metric, ascending=False).reset_index(drop=True)
    ranked_df.insert(0, 'rank', range(1, len(ranked_df) + 1))
    return ranked_df

def generate_confusion_insights(confusion_matrix, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix) if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    recalls = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        total_actual = confusion_matrix[i, :].sum()
        recall = tp / total_actual if total_actual > 0 else 0
        recalls.append((class_names[i], recall))
    recalls.sort(key=lambda x: x[1], reverse=True)
    best = recalls[0] if recalls else ("N/A", 0)
    worst = recalls[-1] if recalls else ("N/A", 0)
    cm_copy = confusion_matrix.copy()
    np.fill_diagonal(cm_copy, 0)
    confusion_msg = ""
    if cm_copy.max() > 0:
        idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        confusion_msg = f"{class_names[idx[0]]} → {class_names[idx[1]]}"
    return {
        'accuracy': f"{accuracy:.1%}",
        'best_class': f"{best[0]} ({best[1]:.1%})",
        'worst_class': f"{worst[0]} ({worst[1]:.1%})",
        'confusion': confusion_msg
    }

# Page UI
def show_comparison_page():
    st.title("Model Comparison Dashboard")
    st.markdown("Compare trained models and analyze their performance")

    # Keep styling consistent with upload/EDA
    st.markdown("""
    <style>
    .stButton>button { background-color: #1976D2; color: white; }
    .highlight { font-size: 14px; color: #87CEEB; font-weight:700; }
    .small-muted { font-size:12px; color:#B0B0B0; }
    </style>
    """, unsafe_allow_html=True)

    # Pre-checks
    missing = []
    if 'trained_models' not in st.session_state:
        missing.append("Trained Models")
    if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
        missing.append("Test Data")
    if missing:
        st.warning(f"Missing: {', '.join(missing)} — complete training & evaluation first")
        cols = st.columns(2)
        with cols[0]:
            if st.button("Go to Training"):
                st.switch_page("pages/training_and_optimization.py")
        with cols[1]:
            if st.button("Upload Data / EDA"):
                st.switch_page("pages/upload_and_eda.py")
        return

    trained_models = st.session_state.trained_models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    st.success(f"Found {len(trained_models)} trained models")

    # Evaluate all models button
    if st.button("Evaluate All Models", type="primary", use_container_width=True):
        with st.spinner("Evaluating models..."):
            results_df, confusion_figs = evaluate_all_models(trained_models, X_test, y_test)
            st.session_state.evaluation_metrics = results_df
            st.session_state.confusion_matrices = confusion_figs
            st.success("Evaluation complete")

    if 'evaluation_metrics' not in st.session_state:
        st.info("Click 'Evaluate All Models' to compute metrics")
        return

    results_df = st.session_state.evaluation_metrics
    confusion_figs = st.session_state.get('confusion_matrices', {})

    st.markdown("---")
    st.subheader("Performance Overview")
    st.markdown("<div class='small-muted'>A single grouped view shows accuracy, F1, precision and recall together to avoid redundant charts.</div>", unsafe_allow_html=True)
    st.plotly_chart(create_overall_comparison_chart(results_df), use_container_width=True)

    # Training time is complementary and kept
    tt_chart = create_training_time_chart(results_df)
    if tt_chart is not None:
        with st.expander("Show Training Time Chart", expanded=False):
            st.plotly_chart(tt_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Rankings & Top Models")
    # Rank by selected metric but show only condensed table and highlighted metrics
    metric_options = ['f1_weighted', 'accuracy', 'precision_weighted', 'recall_weighted', 'roc_auc']
    primary_metric = st.selectbox("Select ranking metric", options=metric_options, index=0)
    ranked_df = rank_models(results_df, primary_metric)

    # Highlight top model prominently
    top = ranked_df.iloc[0]
    st.markdown(f"<div class='highlight'>Top Model: {top['model']} — {primary_metric.replace('_',' ').title()}: {top.get(primary_metric, 0):.4f}</div>", unsafe_allow_html=True)

    # Show compact ranking table (top 10)
    display_df = ranked_df.head(10)[['rank', 'model', primary_metric]].copy()
    display_df[primary_metric] = display_df[primary_metric].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=240)

    st.markdown("**Top 3 Models**")
    for _, row in ranked_df.head(3).iterrows():
        st.markdown(f"- {row['rank']}. **{row['model']}** — {primary_metric.replace('_',' ').title()}: `{row.get(primary_metric, 0):.4f}`")

    st.markdown("---")
    st.subheader("Confusion Matrices & Insights")
    for model_name, fig in confusion_figs.items():
        with st.expander(model_name, expanded=False):
            st.plotly_chart(fig, use_container_width=True)
            try:
                cm_data = np.array(fig.data[0].z)
                insights = generate_confusion_insights(cm_data)
                ins_df = pd.DataFrame([
                    {"Metric": "Overall Accuracy", "Value": insights['accuracy']},
                    {"Metric": "Best Class (recall)", "Value": insights['best_class']},
                    {"Metric": "Worst Class (recall)", "Value": insights['worst_class']},
                    {"Metric": "Common Error", "Value": insights['confusion'] or "N/A"}
                ])
                st.table(ins_df)
            except Exception:
                st.info("Could not parse confusion matrix data for insights")

    st.markdown("---")
    st.subheader("Export Results")
    cols = st.columns(1)
    with cols[0]:
        csv = ranked_df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="model_comparison.csv", mime="text/csv", use_container_width=True)
    st.markdown("---")
    # Navigation
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Back to Training", use_container_width=True):
            st.switch_page("pages/training_and_optimization.py")
    with nav2:
        if st.button("Generate Report", type="primary", use_container_width=True):
            st.session_state['last_viewed_comparison'] = True
            st.switch_page("pages/final_report.py")

if __name__ == "__main__":
    show_comparison_page()
