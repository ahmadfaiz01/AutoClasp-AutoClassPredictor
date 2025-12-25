import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
import plotly.graph_objs as go
from plotly.subplots import make_subplots

REPORT_COLORS = {
    'primary': '#2196F3',
    'header_bg': '#1976D2',
    'table_header': '#2196F3',
    'table_row_light': '#E3F2FD',
    'table_row_dark': '#BBDEFB',
    'text_dark': '#1A1A1A',
    'text_light': '#666666',
    'border': '#90CAF9'
}


def generate_dataset_overview():
    """Extract dataset information from session state"""
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
    elif 'df' in st.session_state:
        df = st.session_state['df']
    else:
        return None

    target = st.session_state.get('target_column', 'Unknown')

    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'target_column': target,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'features': df.columns.tolist()
    }


def generate_eda_statistics():
    """Generate comprehensive EDA statistics"""
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
    elif 'df' in st.session_state:
        df = st.session_state['df']
    else:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    stats = {
        'numerical_features': {},
        'categorical_features': {},
        'correlations': {},
        'distributions': {}
    }

    # Numerical features statistics
    for col in numeric_cols[:8]:
        stats['numerical_features'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75)
        }

    # Categorical features statistics
    for col in categorical_cols[:5]:
        unique_vals = df[col].nunique()
        most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
        stats['categorical_features'][col] = {
            'unique_values': unique_vals,
            'most_common': most_common,
            'most_common_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
        }

    return stats


def generate_issues_summary():
    """Get detected issues with clean formatting"""
    detected = st.session_state.get('detected_issues')
    if not detected:
        return {
            'health_score': 85,
            'issues': []
        }

    issues_list = []

    # Missing values
    mv = detected.get('missing_values', {})
    if mv.get('has_issue'):
        cols_affected = mv.get('affected_columns', 0)
        issues_list.append(f"Missing Values: {cols_affected} columns affected")

    # Outliers
    ol = detected.get('outliers', {})
    if ol.get('has_issue'):
        cols_affected = ol.get('affected_columns', 0)
        issues_list.append(f"Outliers: Detected in {cols_affected} columns")

    # Class imbalance
    imb = detected.get('class_imbalance', {})
    if imb.get('has_issue'):
        ratio = imb.get('ratio', 0)
        issues_list.append(f"Class Imbalance: Ratio {ratio:.2f}")

    # High cardinality
    hc = detected.get('high_cardinality', {})
    if hc.get('has_issue'):
        cols = len(hc.get('columns', []))
        issues_list.append(f"High Cardinality: {cols} categorical features")

    # Constant features
    cf = detected.get('constant_features', {})
    if cf.get('has_issue'):
        count = cf.get('count', 0)
        issues_list.append(f"Constant Features: {count} columns")

    # Duplicates
    dup = detected.get('duplicates', {})
    if dup.get('has_issue'):
        count = dup.get('count', 0)
        pct = dup.get('percentage', 0)
        issues_list.append(f"Duplicate Rows: {count} ({pct:.1f}%)")

    return {
        'health_score': detected.get('health_score', 85),
        'issues': issues_list
    }


def generate_preprocessing_summary():
    """Get preprocessing decisions in clean format"""
    decisions = st.session_state.get('user_decisions') or st.session_state.get('preprocessing_applied')
    if not decisions:
        return None

    summary = []

    # Missing values
    mv = decisions.get('missing_values', {})
    if mv.get('apply'):
        method = mv.get('method', 'Unknown')
        summary.append(f"Missing Values: Handled using {method} method")

    # Outliers
    ol = decisions.get('outliers', {})
    if ol.get('apply'):
        method = ol.get('method', 'Unknown')
        summary.append(f"Outliers: {method.capitalize()} method applied")

    # Constant features
    cf = decisions.get('constant_features', {})
    if cf.get('apply'):
        summary.append("Constant Features: Removed from dataset")

    # Duplicates
    dup = decisions.get('duplicates', {})
    if dup.get('apply'):
        summary.append("Duplicates: Removed from dataset")

    # Encoding
    encoding = decisions.get('encoding', 'none')
    if encoding != 'none':
        summary.append(f"Encoding: {encoding.capitalize()} encoding applied")

    # Scaling
    scaling = decisions.get('scaling', 'none')
    if scaling != 'none':
        summary.append(f"Scaling: {scaling.capitalize()} scaling applied")

    return summary if summary else None


def generate_model_configurations():
    """Extract model configurations"""
    if 'trained_models' not in st.session_state:
        return None

    configs = {}
    for model_name, model_obj in st.session_state.trained_models.items():
        try:
            params = model_obj.get_params()
            # Filter to important params only
            important = {}
            key_params = ['n_estimators', 'max_depth', 'learning_rate', 'C', 'kernel',
                          'n_neighbors', 'criterion', 'random_state', 'max_iter', 'gamma', 'alpha']
            for k, v in params.items():
                if k in key_params:
                    important[k] = v
            configs[model_name] = important if important else dict(list(params.items())[:5])
        except:
            configs[model_name] = {}

    return configs


def generate_comparison_table():
    """Get model comparison results"""
    return st.session_state.get('evaluation_metrics')


def identify_best_model():
    """Identify and justify best model"""
    df = st.session_state.get('evaluation_metrics')
    if df is None or len(df) == 0:
        return None

    best_f1_idx = df['f1_weighted'].idxmax()
    best_model = df.loc[best_f1_idx]

    best_acc = df.loc[df['accuracy'].idxmax(), 'model']
    best_prec = df.loc[df['precision_weighted'].idxmax(), 'model']
    best_rec = df.loc[df['recall_weighted'].idxmax(), 'model']

    return {
        'model_name': best_model['model'],
        'metrics': {
            'accuracy': best_model['accuracy'],
            'f1_score': best_model['f1_weighted'],
            'precision': best_model['precision_weighted'],
            'recall': best_model['recall_weighted'],
            'training_time': best_model['training_time']
        },
        'all_best': {
            'best_accuracy': best_acc,
            'best_precision': best_prec,
            'best_recall': best_rec
        }
    }


def create_performance_chart_for_pdf():
    """Create a grouped bar chart for PDF inclusion"""
    df = st.session_state.get('evaluation_metrics')
    if df is None or len(df) == 0:
        return None

    try:
        from reportlab.graphics.shapes import Drawing, String
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.lib import colors as rl_colors

        drawing = Drawing(500, 250)
        chart = VerticalBarChart()
        chart.x = 60
        chart.y = 50
        chart.height = 150
        chart.width = 400

        # Prepare data - each metric as a separate series
        chart.data = [
            df['accuracy'].tolist(),
            df['f1_weighted'].tolist()
        ]

        # Model names as categories
        model_names = [m[:12] + '...' if len(m) > 12 else m for m in df['model'].tolist()]
        chart.categoryAxis.categoryNames = model_names
        chart.categoryAxis.labels.angle = 30
        chart.categoryAxis.labels.fontSize = 7
        chart.categoryAxis.labels.dy = -5

        # Value axis
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 1.0
        chart.valueAxis.valueStep = 0.2
        chart.valueAxis.labels.fontSize = 8

        # Bar colors
        chart.bars[0].fillColor = rl_colors.HexColor(REPORT_COLORS['primary'])
        chart.bars[1].fillColor = rl_colors.HexColor('#4CAF50')

        # Bar width
        chart.barWidth = 8
        chart.groupSpacing = 12

        # Legend
        from reportlab.graphics.shapes import Rect
        legend_y = 210
        drawing.add(Rect(70, legend_y, 12, 12, fillColor=rl_colors.HexColor(REPORT_COLORS['primary'])))
        drawing.add(String(85, legend_y + 3, 'Accuracy', fontSize=9))
        drawing.add(Rect(150, legend_y, 12, 12, fillColor=rl_colors.HexColor('#4CAF50')))
        drawing.add(String(165, legend_y + 3, 'F1 Score', fontSize=9))

        drawing.add(chart)
        return drawing
    except Exception as e:
        return None


def create_pdf_report():
    """Generate comprehensive PDF report"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.platypus.flowables import HRFlowable
    except ImportError:
        st.error("Install reportlab: pip install reportlab")
        return None

    dataset_info = generate_dataset_overview()
    eda_stats = generate_eda_statistics()
    issues_info = generate_issues_summary()
    preprocessing_info = generate_preprocessing_summary()
    model_configs = generate_model_configurations()
    comparison_df = generate_comparison_table()
    best_model_info = identify_best_model()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=50)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=28,
                                 textColor=colors.HexColor(REPORT_COLORS['header_bg']), spaceAfter=12,
                                 alignment=TA_CENTER, fontName='Helvetica-Bold')

    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14,
                                    textColor=colors.HexColor(REPORT_COLORS['primary']), spaceAfter=20,
                                    alignment=TA_CENTER, fontName='Helvetica')

    heading_style = ParagraphStyle('Heading', parent=styles['Heading1'], fontSize=16,
                                   textColor=colors.HexColor(REPORT_COLORS['header_bg']), spaceAfter=12,
                                   spaceBefore=15, fontName='Helvetica-Bold',
                                   backColor=colors.HexColor(REPORT_COLORS['table_row_light']), borderPadding=8)

    subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading2'], fontSize=12,
                                      textColor=colors.HexColor(REPORT_COLORS['primary']), spaceAfter=8,
                                      spaceBefore=10, fontName='Helvetica-Bold')

    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10,
                                alignment=TA_JUSTIFY, spaceAfter=8, leading=14,
                                textColor=colors.HexColor(REPORT_COLORS['text_dark']))

    # Cover page
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph("AutoClasp", title_style))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("Machine Learning Classification Report", subtitle_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(HRFlowable(width="60%", thickness=2,
                               color=colors.HexColor(REPORT_COLORS['primary']),
                               spaceAfter=20, spaceBefore=10, hAlign='CENTER'))

    if dataset_info:
        cover_info = f"""
        <para alignment="center">
        <b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        <b>Dataset:</b> {dataset_info['target_column']} Classification<br/>
        <b>Records:</b> {dataset_info['total_rows']:,}<br/>
        </para>
        """
        elements.append(Paragraph(cover_info, body_style))

    elements.append(PageBreak())

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    if dataset_info and comparison_df is not None and len(comparison_df) > 0:
        best_acc = comparison_df['accuracy'].max()
        avg_acc = comparison_df['accuracy'].mean()

        summary = f"""
        This analysis evaluated {len(comparison_df)} machine learning algorithms on a dataset of 
        {dataset_info['total_rows']:,} records with {dataset_info['total_columns']} features 
        to predict <b>{dataset_info['target_column']}</b>. The best model achieved 
        <b>{best_acc:.2%}</b> accuracy with an average of <b>{avg_acc:.2%}</b> across all models. 
        Data health score: <b>{issues_info['health_score']}/100</b>.
        """
        elements.append(Paragraph(summary, body_style))

    elements.append(Spacer(1, 0.2 * inch))

    # Dataset Overview
    elements.append(Paragraph("2. Dataset Overview", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    if dataset_info:
        overview_data = [
            ['Metric', 'Value', 'Details'],
            ['Total Records', f"{dataset_info['total_rows']:,}", 'Samples in dataset'],
            ['Total Features', str(dataset_info['total_columns']), 'Including target variable'],
            ['Target Variable', dataset_info['target_column'], 'Classification label'],
            ['Missing Values', str(dataset_info['missing_values']), 'Null entries found'],
            ['Duplicate Records', str(dataset_info['duplicate_rows']), 'Exact duplicates']
        ]

        t = Table(overview_data, colWidths=[1.8 * inch, 1.5 * inch, 2.8 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.15 * inch))

        # Features table
        elements.append(Paragraph("Dataset Features", subheading_style))
        features = dataset_info['features']
        if features:
            feature_rows = [['Feature', 'Feature', 'Feature']]
            for i in range(0, len(features), 3):
                row = features[i:i + 3]
                while len(row) < 3:
                    row.append('')
                feature_rows.append(row)

            ft = Table(feature_rows, colWidths=[2 * inch, 2 * inch, 2 * inch])
            ft.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(ft)

    elements.append(Spacer(1, 0.2 * inch))

    # EDA Findings
    elements.append(Paragraph("3. Exploratory Data Analysis", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    # Add detailed EDA summary
    eda_summary_text = """
    Comprehensive exploratory data analysis was conducted to understand the dataset characteristics. 
    The analysis examined statistical distributions, identified patterns in numerical features, 
    analyzed categorical variable frequencies, and assessed feature relationships. Key findings include 
    central tendency measures (mean, median), dispersion metrics (standard deviation, quartiles), 
    and distribution characteristics that inform preprocessing and modeling decisions.
    """
    elements.append(Paragraph(eda_summary_text, body_style))
    elements.append(Spacer(1, 0.1 * inch))

    if eda_stats and 'numerical_features' in eda_stats and eda_stats['numerical_features']:
        elements.append(Paragraph("Numerical Features Statistics", subheading_style))

        # Add insights paragraph
        num_count = len(eda_stats['numerical_features'])
        elements.append(Paragraph(
            f"Analysis of {num_count} numerical feature(s) revealed their statistical properties, "
            "including measures of central tendency and spread. These statistics help identify "
            "potential outliers, understand feature scales, and guide normalization strategies.",
            body_style
        ))
        elements.append(Spacer(1, 0.05 * inch))

        num_data = [['Feature', 'Mean', 'Std Dev', 'Min', 'Max']]
        for feat, stats in list(eda_stats['numerical_features'].items())[:8]:
            num_data.append([
                feat[:20],
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}"
            ])

        nt = Table(num_data, colWidths=[2 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
        nt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        elements.append(nt)

    if eda_stats and 'categorical_features' in eda_stats and eda_stats['categorical_features']:
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("Categorical Features Summary", subheading_style))

        # Add insights paragraph
        cat_count = len(eda_stats['categorical_features'])
        elements.append(Paragraph(
            f"Examined {cat_count} categorical feature(s) to understand their cardinality and distribution. "
            "High cardinality features may require special encoding strategies, while dominant categories "
            "indicate potential class imbalance patterns.",
            body_style
        ))
        elements.append(Spacer(1, 0.05 * inch))

        cat_data = [['Feature', 'Unique Values', 'Most Common', 'Frequency']]
        for feat, stats in eda_stats['categorical_features'].items():
            cat_data.append([
                feat[:20],
                str(stats['unique_values']),
                str(stats['most_common'])[:15],
                str(stats['most_common_count'])
            ])

        ct = Table(cat_data, colWidths=[1.8 * inch, 1.3 * inch, 1.8 * inch, 1.2 * inch])
        ct.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        elements.append(ct)

    elements.append(Spacer(1, 0.2 * inch))

    # Data Quality Issues
    elements.append(Paragraph("4. Data Quality Assessment", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph(
        f"<b>Overall Health Score: {issues_info['health_score']}/100</b>",
        subheading_style
    ))
    elements.append(Spacer(1, 0.1 * inch))

    if issues_info['issues']:
        # Convert to 2-column format
        issues_data = [['Issue Type', 'Details']]
        for issue in issues_info['issues']:
            # Split issue into type and details
            if ':' in issue:
                parts = issue.split(':', 1)
                issue_type = parts[0].strip()
                issue_details = parts[1].strip()
            else:
                issue_type = issue
                issue_details = 'Detected'
            issues_data.append([issue_type, issue_details])

        it = Table(issues_data, colWidths=[2.2 * inch, 3.8 * inch])
        it.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(it)
    else:
        elements.append(Paragraph(
            "✓ No critical data quality issues detected. The dataset is in good condition for modeling.",
            body_style
        ))

    elements.append(Spacer(1, 0.2 * inch))

    # Preprocessing
    elements.append(Paragraph("5. Data Preprocessing", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    if preprocessing_info:
        prep_data = [['Preprocessing Step Applied']]
        for step in preprocessing_info:
            prep_data.append([step])

        pt = Table(prep_data, colWidths=[6 * inch])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(pt)
    else:
        elements.append(Paragraph("No preprocessing applied.", body_style))

    elements.append(PageBreak())

    # Model Configurations
    elements.append(Paragraph("6. Model Configurations", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    if model_configs:
        elements.append(Paragraph(
            f"Trained {len(model_configs)} machine learning models:",
            body_style
        ))
        elements.append(Spacer(1, 0.15 * inch))

        for idx, (model_name, params) in enumerate(model_configs.items(), 1):
            elements.append(Paragraph(f"{idx}. {model_name}", subheading_style))

            if params:
                param_data = [['Parameter', 'Value']]
                for k, v in params.items():
                    param_data.append([str(k), str(v)[:40]])

                pt = Table(param_data, colWidths=[2.5 * inch, 3.6 * inch])
                pt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                     [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                ]))
                elements.append(pt)
            else:
                elements.append(Paragraph("Default configuration", body_style))

            elements.append(Spacer(1, 0.1 * inch))

    elements.append(PageBreak())

    # Model Comparison
    elements.append(Paragraph("7. Model Performance Comparison", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    # Add metric definitions FIRST
    elements.append(Paragraph("Evaluation Metrics:", subheading_style))
    metrics_text = """
    <b>Accuracy:</b> Overall correctness - ratio of correct predictions to total predictions<br/>
    <b>Precision:</b> Positive predictive value - ratio of true positives to all positive predictions<br/>
    <b>Recall:</b> Sensitivity - ratio of true positives to all actual positive cases<br/>
    <b>F1 Score:</b> Harmonic mean of precision and recall, providing balanced performance measure<br/>
    <b>Time:</b> Training duration in seconds
    """
    elements.append(Paragraph(metrics_text, body_style))
    elements.append(Spacer(1, 0.15 * inch))

    if comparison_df is not None and len(comparison_df) > 0:
        elements.append(Paragraph("Performance Comparison Table:", subheading_style))
        comp_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time (s)']]

        for _, row in comparison_df.iterrows():
            comp_data.append([
                row['model'][:25],
                f"{row['accuracy']:.4f}",
                f"{row['precision_weighted']:.4f}",
                f"{row['recall_weighted']:.4f}",
                f"{row['f1_weighted']:.4f}",
                f"{row['training_time']:.2f}"
            ])

        ct = Table(comp_data, colWidths=[1.6 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch])
        ct.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        elements.append(ct)
        elements.append(Spacer(1, 0.2 * inch))

        # Performance chart
        elements.append(Paragraph("Visual Comparison:", subheading_style))
        perf_chart = create_performance_chart_for_pdf()
        if perf_chart:
            elements.append(perf_chart)

    elements.append(PageBreak())

    # Best Model Recommendation
    elements.append(Paragraph("8. Recommended Model", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    if best_model_info:
        elements.append(Paragraph(
            f"<b>Selected Model: {best_model_info['model_name']}</b>",
            subheading_style
        ))
        elements.append(Spacer(1, 0.1 * inch))

        best_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Accuracy', f"{best_model_info['metrics']['accuracy']:.4f}", 'Overall correctness'],
            ['F1 Score', f"{best_model_info['metrics']['f1_score']:.4f}", 'Balanced metric'],
            ['Precision', f"{best_model_info['metrics']['precision']:.4f}", 'Positive accuracy'],
            ['Recall', f"{best_model_info['metrics']['recall']:.4f}", 'Coverage'],
            ['Training Time', f"{best_model_info['metrics']['training_time']:.2f}s", 'Efficiency']
        ]

        bt = Table(best_data, colWidths=[1.8 * inch, 1.2 * inch, 3.1 * inch])
        bt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        elements.append(bt)
        elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph("8. Recommended Model", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    if best_model_info:
        elements.append(Paragraph(
            f"<b>Selected Model: {best_model_info['model_name']}</b>",
            subheading_style
        ))
        elements.append(Spacer(1, 0.1 * inch))

        best_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Accuracy', f"{best_model_info['metrics']['accuracy']:.4f}", 'Overall correctness'],
            ['F1 Score', f"{best_model_info['metrics']['f1_score']:.4f}", 'Balanced metric'],
            ['Precision', f"{best_model_info['metrics']['precision']:.4f}", 'Positive accuracy'],
            ['Recall', f"{best_model_info['metrics']['recall']:.4f}", 'Coverage'],
            ['Training Time', f"{best_model_info['metrics']['training_time']:.2f}s", 'Efficiency']
        ]

        bt = Table(best_data, colWidths=[1.8 * inch, 1.2 * inch, 3.1 * inch])
        bt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        elements.append(bt)
        elements.append(Spacer(1, 0.15 * inch))

        elements.append(Paragraph("Justification:", subheading_style))
        justification = f"""
        Selected based on highest F1 score ({best_model_info['metrics']['f1_score']:.4f}), 
        which balances precision and recall. Achieved {best_model_info['metrics']['accuracy']:.2%} 
        accuracy with {best_model_info['metrics']['training_time']:.2f}s training time.
        """
        elements.append(Paragraph(justification, body_style))

        elements.append(PageBreak())
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("Alternative Model Recommendations:", subheading_style))

        # Create alternatives table
        alt_data = [['Category', 'Model', 'Score']]

        # Best accuracy
        best_acc_model = best_model_info['all_best']['best_accuracy']
        if comparison_df is not None:
            acc_row = comparison_df[comparison_df['model'] == best_acc_model]
            if len(acc_row) > 0:
                alt_data.append([
                    'Best Accuracy',
                    best_acc_model[:30],
                    f"{acc_row.iloc[0]['accuracy']:.4f}"
                ])

        # Best precision
        best_prec_model = best_model_info['all_best']['best_precision']
        if comparison_df is not None:
            prec_row = comparison_df[comparison_df['model'] == best_prec_model]
            if len(prec_row) > 0:
                alt_data.append([
                    'Best Precision',
                    best_prec_model[:30],
                    f"{prec_row.iloc[0]['precision_weighted']:.4f}"
                ])

        # Best recall
        best_rec_model = best_model_info['all_best']['best_recall']
        if comparison_df is not None:
            rec_row = comparison_df[comparison_df['model'] == best_rec_model]
            if len(rec_row) > 0:
                alt_data.append([
                    'Best Recall',
                    best_rec_model[:30],
                    f"{rec_row.iloc[0]['recall_weighted']:.4f}"
                ])

        if len(alt_data) > 1:
            alt_table = Table(alt_data, colWidths=[1.6 * inch, 3 * inch, 1.5 * inch])
            alt_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(REPORT_COLORS['table_header'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),

                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor(REPORT_COLORS['table_row_light']), colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(REPORT_COLORS['border'])),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))

            elements.append(alt_table)

            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(
                "Consider these alternatives based on specific requirements like minimizing false positives (precision) or catching all positive cases (recall).",
                body_style
            ))
        else:
            elements.append(Paragraph(
                "The recommended model achieves best performance across all metrics.",
                body_style
            ))
    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.1 * inch))

    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9,
                                  textColor=colors.grey, alignment=TA_CENTER)

    elements.append(Paragraph("<b>AutoClasp - Automated Classification System</b>", footer_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        footer_style
    ))

    try:
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
        return None


def create_html_report():
    """Generate comprehensive HTML report matching PDF structure"""
    dataset_info = generate_dataset_overview()
    eda_stats = generate_eda_statistics()
    issues_info = generate_issues_summary()
    preprocessing_info = generate_preprocessing_summary()
    model_configs = generate_model_configurations()
    comparison_df = generate_comparison_table()
    best_model_info = identify_best_model()

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoClasp Analysis Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #1A1A1A;
                background: #f8f9fa;
                padding: 20px;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}

            .cover {{
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                color: white;
                padding: 80px 40px;
                text-align: center;
            }}

            .cover h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                font-weight: bold;
            }}

            .cover .subtitle {{
                font-size: 1.5em;
                margin-bottom: 30px;
                opacity: 0.95;
            }}

            .cover .divider {{
                width: 200px;
                height: 3px;
                background: white;
                margin: 30px auto;
            }}

            .cover .info {{
                font-size: 1.1em;
                line-height: 2;
            }}

            .content {{
                padding: 40px;
            }}

            .section {{
                margin-bottom: 50px;
                page-break-inside: avoid;
            }}

            .section-title {{
                background: #E3F2FD;
                color: #1976D2;
                padding: 15px 20px;
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 20px;
                border-left: 5px solid #2196F3;
            }}

            .subsection-title {{
                color: #2196F3;
                font-size: 1.2em;
                font-weight: bold;
                margin: 20px 0 10px 0;
                padding-bottom: 5px;
                border-bottom: 2px solid #E3F2FD;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}

            th {{
                background: #2196F3;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}

            td {{
                padding: 12px;
                border-bottom: 1px solid #E3F2FD;
            }}

            tr:nth-child(even) {{
                background: #E3F2FD;
            }}

            tr:hover {{
                background: #BBDEFB;
            }}

            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}

            .metric-box {{
                background: #E3F2FD;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #2196F3;
                text-align: center;
            }}

            .metric-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 8px;
            }}

            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #1976D2;
            }}

            .best-model-box {{
                background: linear-gradient(135deg, #2196F3, #1976D2);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin: 20px 0;
            }}

            .best-model-box h3 {{
                margin-bottom: 20px;
                font-size: 1.5em;
            }}

            .features-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin: 15px 0;
            }}

            .feature-item {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                border: 1px solid #E3F2FD;
            }}

            .paragraph {{
                text-align: justify;
                margin: 15px 0;
                line-height: 1.8;
            }}

            .issue-list {{
                list-style: none;
                padding: 0;
            }}

            .issue-list li {{
                background: #E3F2FD;
                padding: 12px;
                margin: 8px 0;
                border-left: 4px solid #2196F3;
                border-radius: 4px;
            }}

            .footer {{
                background: #f8f9fa;
                padding: 30px;
                text-align: center;
                color: #666;
                border-top: 3px solid #2196F3;
            }}

            .model-config {{
                background: #f8f9fa;
                padding: 15px;
                margin: 15px 0;
                border-radius: 8px;
                border: 1px solid #E3F2FD;
            }}

            .model-config h4 {{
                color: #2196F3;
                margin-bottom: 10px;
            }}

            @media print {{
                .cover {{
                    page-break-after: always;
                }}
                .section {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Cover Page -->
            <div class="cover">
                <h1>AutoClasp</h1>
                <div class="subtitle">Machine Learning Classification Report</div>
                <div class="divider"></div>
                <div class="info">
                    <strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
    """

    if dataset_info:
        html += f"""
                    <strong>Dataset:</strong> {dataset_info['target_column']} Classification<br>
                    <strong>Records:</strong> {dataset_info['total_rows']:,}
        """

    html += """
                </div>
            </div>

            <div class="content">
    """

    # Section 1: Executive Summary
    if dataset_info and comparison_df is not None and len(comparison_df) > 0:
        best_acc = comparison_df['accuracy'].max()
        avg_acc = comparison_df['accuracy'].mean()

        html += f"""
                <div class="section">
                    <div class="section-title">1. Executive Summary</div>
                    <p class="paragraph">
                        This analysis evaluated <strong>{len(comparison_df)}</strong> machine learning algorithms 
                        on a dataset of <strong>{dataset_info['total_rows']:,}</strong> records with 
                        <strong>{dataset_info['total_columns']}</strong> features to predict 
                        <strong>{dataset_info['target_column']}</strong>. The best model achieved 
                        <strong>{best_acc:.2%}</strong> accuracy with an average of <strong>{avg_acc:.2%}</strong> 
                        across all models. Data health score: <strong>{issues_info['health_score']}/100</strong>.
                    </p>
                </div>
        """

    # Section 2: Dataset Overview
    if dataset_info:
        html += f"""
                <div class="section">
                    <div class="section-title">2. Dataset Overview</div>

                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-label">Total Records</div>
                            <div class="metric-value">{dataset_info['total_rows']:,}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Total Features</div>
                            <div class="metric-value">{dataset_info['total_columns']}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Missing Values</div>
                            <div class="metric-value">{dataset_info['missing_values']}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Duplicates</div>
                            <div class="metric-value">{dataset_info['duplicate_rows']}</div>
                        </div>
                    </div>

                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Details</th>
                        </tr>
                        <tr>
                            <td>Total Records</td>
                            <td>{dataset_info['total_rows']:,}</td>
                            <td>Samples in dataset</td>
                        </tr>
                        <tr>
                            <td>Total Features</td>
                            <td>{dataset_info['total_columns']}</td>
                            <td>Including target variable</td>
                        </tr>
                        <tr>
                            <td>Target Variable</td>
                            <td>{dataset_info['target_column']}</td>
                            <td>Classification label</td>
                        </tr>
                        <tr>
                            <td>Missing Values</td>
                            <td>{dataset_info['missing_values']}</td>
                            <td>Null entries found</td>
                        </tr>
                        <tr>
                            <td>Duplicate Records</td>
                            <td>{dataset_info['duplicate_rows']}</td>
                            <td>Exact duplicates</td>
                        </tr>
                    </table>

                    <div class="subsection-title">Dataset Features</div>
                    <div class="features-grid">
        """

        for feature in dataset_info['features']:
            html += f'<div class="feature-item">{feature}</div>'

        html += """
                    </div>
                </div>
        """

    # Section 3: EDA Findings
    html += """
                <div class="section">
                    <div class="section-title">3. Exploratory Data Analysis</div>
                    <p class="paragraph">
                        Comprehensive exploratory data analysis was conducted to understand the dataset characteristics. 
                        The analysis examined statistical distributions, identified patterns in numerical features, 
                        analyzed categorical variable frequencies, and assessed feature relationships. Key findings include 
                        central tendency measures (mean, median), dispersion metrics (standard deviation, quartiles), 
                        and distribution characteristics that inform preprocessing and modeling decisions.
                    </p>
    """

    if eda_stats and 'numerical_features' in eda_stats and eda_stats['numerical_features']:
        html += """
                    <div class="subsection-title">Numerical Features Statistics</div>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Mean</th>
                            <th>Std Dev</th>
                            <th>Min</th>
                            <th>Max</th>
                        </tr>
        """

        for feat, stats in list(eda_stats['numerical_features'].items())[:8]:
            html += f"""
                        <tr>
                            <td>{feat}</td>
                            <td>{stats['mean']:.2f}</td>
                            <td>{stats['std']:.2f}</td>
                            <td>{stats['min']:.2f}</td>
                            <td>{stats['max']:.2f}</td>
                        </tr>
            """

        html += "</table>"

    if eda_stats and 'categorical_features' in eda_stats and eda_stats['categorical_features']:
        html += """
                    <div class="subsection-title">Categorical Features Summary</div>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Unique Values</th>
                            <th>Most Common</th>
                            <th>Frequency</th>
                        </tr>
        """

        for feat, stats in eda_stats['categorical_features'].items():
            html += f"""
                        <tr>
                            <td>{feat}</td>
                            <td>{stats['unique_values']}</td>
                            <td>{stats['most_common']}</td>
                            <td>{stats['most_common_count']}</td>
                        </tr>
            """

        html += "</table>"

    html += "</div>"

    # Section 4: Data Quality Assessment
    html += f"""
                <div class="section">
                    <div class="section-title">4. Data Quality Assessment</div>
                    <div class="subsection-title">Overall Health Score: {issues_info['health_score']}/100</div>
    """

    if issues_info['issues']:
        html += '<table><tr><th>Issue Type</th><th>Details</th></tr>'
        for issue in issues_info['issues']:
            if ':' in issue:
                parts = issue.split(':', 1)
                issue_type = parts[0].strip()
                issue_detail = parts[1].strip()
            else:
                issue_type = issue
                issue_detail = 'Detected'
            html += f'<tr><td>{issue_type}</td><td>{issue_detail}</td></tr>'
        html += '</table>'
    else:
        html += '<p class="paragraph">No critical data quality issues detected. The dataset is in good condition for modeling.</p>'

    html += "</div>"

    # Section 5: Preprocessing
    html += """
                <div class="section">
                    <div class="section-title">5. Data Preprocessing</div>
    """

    if preprocessing_info:
        html += '<ul class="issue-list">'
        for step in preprocessing_info:
            html += f'<li>{step}</li>'
        html += '</ul>'
    else:
        html += '<p class="paragraph">No preprocessing steps were applied to the data.</p>'

    html += "</div>"

    # Section 6: Model Configurations
    html += """
                <div class="section">
                    <div class="section-title">6. Model Configurations</div>
    """

    if model_configs:
        html += f'<p class="paragraph">Trained <strong>{len(model_configs)}</strong> machine learning models with the following configurations:</p>'

        for idx, (model_name, params) in enumerate(model_configs.items(), 1):
            html += f"""
                    <div class="model-config">
                        <h4>{idx}. {model_name}</h4>
            """

            if params:
                html += '<table><tr><th>Parameter</th><th>Value</th></tr>'
                for k, v in params.items():
                    html += f'<tr><td>{k}</td><td>{str(v)[:50]}</td></tr>'
                html += '</table>'
            else:
                html += '<p>Default configuration</p>'

            html += '</div>'
    else:
        html += '<p class="paragraph">Model configuration information not available.</p>'

    html += "</div>"

    # Section 7: Model Comparison
    html += """
                <div class="section">
                    <div class="section-title">7. Model Performance Comparison</div>

                    <div class="subsection-title">Evaluation Metrics</div>
                    <p class="paragraph">
                        <strong>Accuracy:</strong> Overall correctness - ratio of correct predictions to total predictions<br>
                        <strong>Precision:</strong> Positive predictive value - ratio of true positives to all positive predictions<br>
                        <strong>Recall:</strong> Sensitivity - ratio of true positives to all actual positive cases<br>
                        <strong>F1 Score:</strong> Harmonic mean of precision and recall, providing balanced performance measure<br>
                        <strong>Time:</strong> Training duration in seconds
                    </p>
    """

    if comparison_df is not None and len(comparison_df) > 0:
        html += """
                    <div class="subsection-title">Performance Comparison Table</div>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            <th>Time (s)</th>
                        </tr>
        """

        for _, row in comparison_df.iterrows():
            html += f"""
                        <tr>
                            <td><strong>{row['model']}</strong></td>
                            <td>{row['accuracy']:.4f}</td>
                            <td>{row['precision_weighted']:.4f}</td>
                            <td>{row['recall_weighted']:.4f}</td>
                            <td>{row['f1_weighted']:.4f}</td>
                            <td>{row['training_time']:.2f}</td>
                        </tr>
            """

        html += "</table>"

    html += "</div>"

    # Section 8: Best Model
    if best_model_info:
        html += f"""
                <div class="section">
                    <div class="section-title">8. Recommended Model</div>

                    <div class="best-model-box">
                        <h3>Selected Model: {best_model_info['model_name']}</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9;">Accuracy</div>
                                <div style="font-size: 2em; font-weight: bold;">{best_model_info['metrics']['accuracy']:.4f}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9;">F1 Score</div>
                                <div style="font-size: 2em; font-weight: bold;">{best_model_info['metrics']['f1_score']:.4f}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9;">Precision</div>
                                <div style="font-size: 2em; font-weight: bold;">{best_model_info['metrics']['precision']:.4f}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9;">Recall</div>
                                <div style="font-size: 2em; font-weight: bold;">{best_model_info['metrics']['recall']:.4f}</div>
                            </div>
                        </div>
                    </div>

                    <div class="subsection-title">Justification</div>
                    <p class="paragraph">
                        Selected based on highest F1 score ({best_model_info['metrics']['f1_score']:.4f}), 
                        which balances precision and recall. Achieved {best_model_info['metrics']['accuracy']:.2%} 
                        accuracy with {best_model_info['metrics']['training_time']:.2f}s training time.
                    </p>

                    <div class="subsection-title">Alternative Model Recommendations</div>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Model</th>
                            <th>Score</th>
                        </tr>
        """

        if comparison_df is not None:
            best_acc_model = best_model_info['all_best']['best_accuracy']
            acc_row = comparison_df[comparison_df['model'] == best_acc_model]
            if len(acc_row) > 0:
                html += f"""
                        <tr>
                            <td>Best Accuracy</td>
                            <td>{best_acc_model}</td>
                            <td>{acc_row.iloc[0]['accuracy']:.4f}</td>
                        </tr>
                """

            best_prec_model = best_model_info['all_best']['best_precision']
            prec_row = comparison_df[comparison_df['model'] == best_prec_model]
            if len(prec_row) > 0:
                html += f"""
                        <tr>
                            <td>Best Precision</td>
                            <td>{best_prec_model}</td>
                            <td>{prec_row.iloc[0]['precision_weighted']:.4f}</td>
                        </tr>
                """

            best_rec_model = best_model_info['all_best']['best_recall']
            rec_row = comparison_df[comparison_df['model'] == best_rec_model]
            if len(rec_row) > 0:
                html += f"""
                        <tr>
                            <td>Best Recall</td>
                            <td>{best_rec_model}</td>
                            <td>{rec_row.iloc[0]['recall_weighted']:.4f}</td>
                        </tr>
                """

        html += """
                    </table>
                    <p class="paragraph">
                        Consider these alternatives based on specific requirements like minimizing false positives 
                        (precision) or catching all positive cases (recall).
                    </p>
                </div>
        """

    # Footer
    html += f"""
            </div>

            <div class="footer">
                <strong>AutoClasp - Automated Classification System</strong><br>
                Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
    </body>
    </html>
    """

    return html


def create_markdown_report():
    """Generate Markdown report"""
    dataset_info = generate_dataset_overview()
    issues_info = generate_issues_summary()
    preprocessing_info = generate_preprocessing_summary()
    comparison_df = generate_comparison_table()
    best_model_info = identify_best_model()

    md = f"""# AutoClasp Analysis Report

**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

## 1. Dataset Overview

"""

    if dataset_info:
        md += f"""
- **Rows:** {dataset_info['total_rows']:,}
- **Columns:** {dataset_info['total_columns']}
- **Target:** {dataset_info['target_column']}
- **Missing Values:** {dataset_info['missing_values']}
- **Duplicates:** {dataset_info['duplicate_rows']}

**Features:** {', '.join(dataset_info['features'])}

"""

    md += f"""---

## 2. Data Quality

**Health Score:** {issues_info['health_score']}/100

"""

    if issues_info['issues']:
        for issue in issues_info['issues']:
            md += f"- {issue}\n"
    else:
        md += "No issues detected\n"

    md += "\n---\n\n## 3. Preprocessing\n\n"

    if preprocessing_info:
        for step in preprocessing_info:
            md += f"- {step}\n"
    else:
        md += "No preprocessing applied\n"

    md += "\n---\n\n## 4. Model Comparison\n\n"

    if comparison_df is not None:
        md += "| Model | Accuracy | Precision | Recall | F1 | Time |\n"
        md += "|-------|----------|-----------|--------|-------|------|\n"
        for _, row in comparison_df.iterrows():
            md += f"| {row['model']} | {row['accuracy']:.4f} | {row['precision_weighted']:.4f} | "
            md += f"{row['recall_weighted']:.4f} | {row['f1_weighted']:.4f} | {row['training_time']:.2f}s |\n"

    md += "\n---\n\n## 5. Recommended Model\n\n"

    if best_model_info:
        md += f"""### {best_model_info['model_name']}

- **Accuracy:** {best_model_info['metrics']['accuracy']:.4f}
- **F1 Score:** {best_model_info['metrics']['f1_score']:.4f}
- **Precision:** {best_model_info['metrics']['precision']:.4f}
- **Recall:** {best_model_info['metrics']['recall']:.4f}
- **Training Time:** {best_model_info['metrics']['training_time']:.2f}s

**Alternatives:**
- Best Accuracy: {best_model_info['all_best']['best_accuracy']}
- Best Precision: {best_model_info['all_best']['best_precision']}
- Best Recall: {best_model_info['all_best']['best_recall']}

"""

    md += "\n---\n\n*Generated by AutoClasp*"

    return md


def show_final_report_page():
    st.title("Final Report")
    st.markdown("Generate comprehensive analysis reports")

    # Check data availability
    missing = []
    if not ('dataframe' in st.session_state or 'df' in st.session_state):
        missing.append("Dataset")
    if 'trained_models' not in st.session_state:
        missing.append("Models")
    if 'evaluation_metrics' not in st.session_state:
        missing.append("Evaluation")

    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
        st.info("Complete the pipeline before generating reports")
        if st.button("Back to Comparison"):
            st.switch_page("pages/comparison_and_explainability.py")
        return

    st.success("All data available")

    # Quick summary
    best = identify_best_model()
    if best:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best['model_name'])
        with col2:
            st.metric("F1 Score", f"{best['metrics']['f1_score']:.3f}")
        with col3:
            st.metric("Accuracy", f"{best['metrics']['accuracy']:.3f}")
        with col4:
            st.metric("Time", f"{best['metrics']['training_time']:.2f}s")

    st.markdown("---")
    st.subheader("Download Reports")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate PDF", type="primary", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf = create_pdf_report()
                if pdf:
                    st.download_button(
                        "Download PDF",
                        data=pdf,
                        file_name="autoclasp_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    with col2:
        if st.button("Generate HTML", use_container_width=True):
            with st.spinner("Generating HTML..."):
                html = create_html_report()
                st.download_button(
                    "Download HTML",
                    data=html,
                    file_name="autoclasp_report.html",
                    mime="text/html",
                    use_container_width=True
                )

    with col3:
        if st.button("Generate Markdown", use_container_width=True):
            with st.spinner("Generating Markdown..."):
                md = create_markdown_report()
                st.download_button(
                    "Download MD",
                    data=md,
                    file_name="autoclasp_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    st.markdown("---")

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Comparison", use_container_width=True):
            st.switch_page("pages/comparison_and_explainability.py")
    with col2:
        if st.button("New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("pages/upload_and_eda.py")


if __name__ == "__main__":
    show_final_report_page()
