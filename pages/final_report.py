# python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
import plotly.graph_objs as go

REPORT_COLORS = {
    'primary': '#2196F3',
    'header_bg': '#1976D2',
    'table_header': '#2196F3',
    'table_row_light': '#E3F2FD',
    'text_dark': '#1A1A1A'
}

def _get_df_from_state():
    if 'dataframe' in st.session_state:
        return st.session_state['dataframe']
    if 'df' in st.session_state:
        return st.session_state['df']
    return None

def generate_dataset_overview():
    df = _get_df_from_state()
    if df is None:
        return None
    target = st.session_state.get('target_column', 'Unknown')
    return {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'target_column': target,
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'features': df.columns.tolist()
    }

def generate_eda_statistics():
    df = _get_df_from_state()
    if df is None:
        return None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    stats = {'numerical_features': {}, 'categorical_features': {}}
    for col in numeric_cols[:8]:
        stats['numerical_features'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    for col in categorical_cols[:8]:
        cnts = df[col].value_counts()
        mode = cnts.index[0] if len(cnts) > 0 else 'N/A'
        stats['categorical_features'][col] = {
            'unique': df[col].nunique(),
            'most_common': mode,
            'count': int(cnts.iloc[0]) if len(cnts) > 0 else 0
        }
    return stats

def generate_issues_summary():
    detected = st.session_state.get('detected_issues', {})
    if not detected:
        return {'health_score': 85, 'issues': []}
    issues = []
    if detected.get('missing_values', {}).get('has_issue'):
        issues.append(f"Missing Values: {detected['missing_values'].get('affected_columns', 0)} columns affected")
    if detected.get('class_imbalance', {}).get('has_issue'):
        r = detected['class_imbalance'].get('ratio', 0)
        issues.append(f"Class Imbalance: Ratio {r:.2f}")
    if detected.get('high_cardinality', {}).get('has_issue'):
        issues.append(f"High Cardinality: {len(detected['high_cardinality'].get('columns', []))} features")
    if detected.get('duplicates', {}).get('has_issue'):
        issues.append(f"Duplicate Rows: {detected['duplicates'].get('count', 0)}")
    return {'health_score': detected.get('health_score', 85), 'issues': issues}

def generate_preprocessing_summary():
    decisions = st.session_state.get('user_decisions') or st.session_state.get('preprocessing_applied')
    if not decisions:
        return []
    summary = []
    mv = decisions.get('missing_values', {})
    if mv.get('apply'):
        summary.append(f"Missing values handled: {mv.get('method', 'imputation')}")
    if decisions.get('encoding'):
        summary.append(f"Encoding: {decisions.get('encoding')}")
    if decisions.get('scaling'):
        summary.append(f"Scaling: {decisions.get('scaling')}")
    if decisions.get('constant_features', {}).get('apply'):
        summary.append("Constant features removed")
    if decisions.get('duplicates', {}).get('apply'):
        summary.append("Duplicate rows removed")
    return summary

def generate_model_configurations():
    if 'trained_models' not in st.session_state:
        return {}
    configs = {}
    for name, obj in st.session_state['trained_models'].items():
        try:
            params = obj.get_params()
            key_params = ['n_estimators', 'max_depth', 'learning_rate', 'C', 'kernel', 'n_neighbors', 'random_state', 'max_iter']
            important = {k: params[k] for k in key_params if k in params}
            configs[name] = important or dict(list(params.items())[:6])
        except Exception:
            configs[name] = {}
    return configs

def generate_comparison_table():
    return st.session_state.get('evaluation_metrics')

def identify_best_model():
    df = generate_comparison_table()
    if df is None or len(df) == 0:
        return None
    best_idx = df['f1_weighted'].idxmax() if 'f1_weighted' in df.columns else df.index[0]
    best = df.loc[best_idx]
    return {
        'model_name': best['model'],
        'metrics': {
            'accuracy': best.get('accuracy', np.nan),
            'f1_score': best.get('f1_weighted', np.nan),
            'precision': best.get('precision_weighted', np.nan),
            'recall': best.get('recall_weighted', np.nan),
            'training_time': best.get('training_time', np.nan)
        },
        'all_best': {
            'best_accuracy': df.loc[df['accuracy'].idxmax()]['model'] if 'accuracy' in df.columns else None,
            'best_precision': df.loc[df['precision_weighted'].idxmax()]['model'] if 'precision_weighted' in df.columns else None,
            'best_recall': df.loc[df['recall_weighted'].idxmax()]['model'] if 'recall_weighted' in df.columns else None
        }
    }

def create_pdf_report():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
    except ImportError:
        st.error("PDF export requires `reportlab`. Install via `pip install reportlab`")
        return None

    dataset_info = generate_dataset_overview()
    eda_stats = generate_eda_statistics()
    issues_info = generate_issues_summary()
    preprocessing_info = generate_preprocessing_summary()
    model_configs = generate_model_configurations()
    comparison_df = generate_comparison_table()
    best_model = identify_best_model()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, textColor=colors.HexColor(REPORT_COLORS['header_bg']))
    normal = styles['Normal']

    elements.append(Paragraph("AutoClasp - Classification Report", title_style))
    elements.append(Spacer(1, 0.1 * inch))

    if dataset_info:
        elements.append(Paragraph(f"Dataset: {dataset_info['target_column']}", normal))
        elements.append(Paragraph(f"Records: {dataset_info['total_rows']:,} — Features: {dataset_info['total_columns']}", normal))
        elements.append(Spacer(1, 0.1 * inch))

    if comparison_df is not None and len(comparison_df) > 0:
        # Simple comparison table in PDF
        header = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'Time(s)']
        data = [header]
        for _, r in comparison_df.iterrows():
            data.append([
                r['model'][:30],
                f"{r.get('accuracy', np.nan):.4f}",
                f"{r.get('f1_weighted', np.nan):.4f}",
                f"{r.get('precision_weighted', np.nan):.4f}",
                f"{r.get('recall_weighted', np.nan):.4f}",
                f"{r.get('training_time', np.nan):.2f}"
            ])
        t = Table(data, colWidths=[2.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

    # Best model summary
    if best_model:
        elements.append(Paragraph(f"Recommended Model: {best_model['model_name']}", normal))
        elements.append(Spacer(1, 0.05 * inch))
        metrics = best_model['metrics']
        for k, v in metrics.items():
            elements.append(Paragraph(f"{k.title()}: {v}", normal))
        elements.append(Spacer(1, 0.1 * inch))

    try:
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def create_html_report():
    dataset_info = generate_dataset_overview()
    comparison_df = generate_comparison_table()
    issues_info = generate_issues_summary()
    preprocessing_info = generate_preprocessing_summary()
    best_model = identify_best_model()

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>AutoClasp Report</title>
    <style>body{{font-family:Segoe UI,Arial;line-height:1.6;color:#1A1A1A;padding:20px}}.h{{color:{REPORT_COLORS['header_bg']}}}</style></head><body>"""
    html += f"<h1 class='h'>AutoClasp - Classification Report</h1>"
    if dataset_info:
        html += f"<p><strong>Dataset:</strong> {dataset_info['target_column']} — <strong>Records:</strong> {dataset_info['total_rows']:,}</p>"
    if comparison_df is not None and len(comparison_df) > 0:
        html += "<h2>Model Comparison</h2><table border='1' cellpadding='6'><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>Time</th></tr>"
        for _, r in comparison_df.iterrows():
            html += f"<tr><td>{r['model']}</td><td>{r.get('accuracy', np.nan):.4f}</td><td>{r.get('f1_weighted', np.nan):.4f}</td><td>{r.get('precision_weighted', np.nan):.4f}</td><td>{r.get('recall_weighted', np.nan):.4f}</td><td>{r.get('training_time', np.nan):.2f}</td></tr>"
        html += "</table>"
    if best_model:
        html += f"<h2>Recommended Model: {best_model['model_name']}</h2>"
        html += "<ul>"
        for k, v in best_model['metrics'].items():
            html += f"<li><strong>{k.title()}:</strong> {v}</li>"
        html += "</ul>"
    html += "</body></html>"
    return html

def create_markdown_report():
    dataset_info = generate_dataset_overview()
    comparison_df = generate_comparison_table()
    best_model = identify_best_model()
    md = f"# AutoClasp Report\n\nGenerated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}\n\n"
    if dataset_info:
        md += f"- Rows: {dataset_info['total_rows']:,}\n- Columns: {dataset_info['total_columns']}\n- Target: {dataset_info['target_column']}\n\n"
    if comparison_df is not None:
        md += "## Model Comparison\n\n| Model | Accuracy | F1 | Precision | Recall | Time |\n|---|---:|---:|---:|---:|---:|\n"
        for _, r in comparison_df.iterrows():
            md += f"| {r['model']} | {r.get('accuracy',np.nan):.4f} | {r.get('f1_weighted',np.nan):.4f} | {r.get('precision_weighted',np.nan):.4f} | {r.get('recall_weighted',np.nan):.4f} | {r.get('training_time',np.nan):.2f}s |\n"
    if best_model:
        md += f"\n## Recommended Model: {best_model['model_name']}\n"
        for k, v in best_model['metrics'].items():
            md += f"- **{k.title()}**: {v}\n"
    return md

def show_final_report_page():
    st.title("Final Report")
    st.markdown("Generate comprehensive analysis reports (PDF / HTML / MD)")

    # Check availability
    missing = []
    if _get_df_from_state() is None:
        missing.append("Dataset")
    if 'trained_models' not in st.session_state:
        missing.append("Models")
    if 'evaluation_metrics' not in st.session_state:
        missing.append("Evaluation")
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
        if st.button("Back to Comparison"):
            st.switch_page("pages/comparison_and_explainability.py")
        return

    st.success("All data available")
    best = identify_best_model()
    if best:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best['model_name'])
        col2.metric("F1 Score", f"{best['metrics']['f1_score']:.3f}")
        col3.metric("Accuracy", f"{best['metrics']['accuracy']:.3f}")
        col4.metric("Time (s)", f"{best['metrics']['training_time']:.2f}")

    st.markdown("---")
    st.subheader("Download Reports")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Preparing PDF..."):
                pdf = create_pdf_report()
                if pdf:
                    st.download_button("Download PDF", data=pdf, file_name="autoclasp_report.pdf", mime="application/pdf", use_container_width=True)
    with c2:
        if st.button("Generate HTML", use_container_width=True):
            with st.spinner("Preparing HTML..."):
                html = create_html_report()
                st.download_button("Download HTML", data=html, file_name="autoclasp_report.html", mime="text/html", use_container_width=True)
    with c3:
        if st.button("Generate Markdown", use_container_width=True):
            with st.spinner("Preparing MD..."):
                md = create_markdown_report()
                st.download_button("Download MD", data=md, file_name="autoclasp_report.md", mime="text/markdown", use_container_width=True)

    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Back to Comparison", use_container_width=True):
            st.switch_page("pages/comparison_and_explainability.py")
    with nav2:
        if st.button("Start New Analysis", use_container_width=True):
            keys = list(st.session_state.keys())
            for k in keys:
                del st.session_state[k]
            st.switch_page("pages/upload_and_eda.py")

if __name__ == "__main__":
    show_final_report_page()
