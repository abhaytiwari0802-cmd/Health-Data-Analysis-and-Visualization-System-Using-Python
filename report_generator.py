"""
Module: report_generator.py
Description: Generates multi-format analytical reports:
             - Text Summary Report
             - Print-Ready HTML Medical Audit Report
             - Pure-Python Standard PDF Report (zero external dependencies)
"""

import os
import io
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


def generate_text_summary_report(
    df: pd.DataFrame,
    audit_log: Dict[str, Any],
    stats_df: pd.DataFrame,
    corr_df: Optional[pd.DataFrame] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    comparison_summary: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs a plain-text academic and analytical summary report.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 78)
    report.append("  HEALTH DATA ANALYSIS AND VISUALIZATION SYSTEM - SUMMARY REPORT")
    report.append("  MCA Minor Project - Master of Computer Applications")
    report.append("=" * 78)
    report.append(f"Generated On       : {timestamp}")
    report.append(f"Total Records      : {len(df)}")
    report.append(f"Total Parameters   : {len(df.columns)}")
    report.append("=" * 78)
    report.append("")
    
    # 1. Cleaning Audit
    report.append("1. DATA QUALITY & CLEANING AUDIT (NumPy Engine)")
    report.append("-" * 55)
    report.append(f"Initial Records Imported : {audit_log.get('initial_rows', 'N/A')}")
    report.append(f"Duplicate Rows Removed   : {audit_log.get('duplicates_removed', 0)}")
    report.append(f"Imputation Strategy      : {audit_log.get('imputation_strategy', 'median').capitalize()} Imputation")
    
    imputed = audit_log.get("imputed_details", {})
    if imputed:
        report.append("Missing Value Imputations:")
        for col, details in imputed.items():
            if isinstance(details, dict):
                report.append(f"  - {col:<18} : {details.get('imputed_count', 0)} values imputed with {details.get('replacement_value', 'N/A')}")
    else:
        report.append("Missing Values           : No missing values detected.")
        
    clipped = audit_log.get("clipped_details", {})
    if clipped:
        report.append("Out-of-Range Corrections (NumPy np.clip):")
        for col, count in clipped.items():
            report.append(f"  - {col:<18} : {count} anomalies bounded to reference limits")
            
    report.append(f"BMI Re-computations      : {audit_log.get('bmi_recalculated_count', 0)} rows updated via vectorized formula")
    report.append(f"Final Clean Records      : {audit_log.get('final_rows', len(df))}")
    report.append("")
    
    # 2. Descriptive Statistics Table
    report.append("2. DESCRIPTIVE STATISTICS (NumPy Calculations)")
    report.append("-" * 78)
    if not stats_df.empty:
        header = f"{'Variable':<16} {'Count':>6} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'Min':>8} {'Max':>8} {'IQR':>8}"
        report.append(header)
        report.append("-" * 78)
        for _, row in stats_df.iterrows():
            line = (
                f"{str(row.get('Variable', '')):<16} "
                f"{int(row.get('Count', 0)):>6} "
                f"{float(row.get('Mean', 0.0)):>8.2f} "
                f"{float(row.get('Median', 0.0)):>8.2f} "
                f"{float(row.get('Std_Dev', 0.0)):>8.2f} "
                f"{float(row.get('Min', 0.0)):>8.2f} "
                f"{float(row.get('Max', 0.0)):>8.2f} "
                f"{float(row.get('IQR', 0.0)):>8.2f}"
            )
            report.append(line)
    report.append("")
    
    # 3. Top Correlations
    if corr_df is not None and not corr_df.empty:
        report.append("3. KEY BIOMEDICAL CORRELATIONS")
        report.append("-" * 55)
        pairs = []
        cols = corr_df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val = corr_df.iloc[i, j]
                pairs.append((cols[i], cols[j], r_val))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for x_var, y_var, r in pairs[:6]:
            direction = "Positive" if r > 0 else "Negative"
            strength = "Strong" if abs(r) >= 0.7 else ("Moderate" if abs(r) >= 0.4 else "Weak")
            report.append(f"  * {x_var} vs {y_var:<18}: r = {r:+.3f} ({strength} {direction})")
        report.append("  Note: Statistical correlation indicates association, not medical causation.")
    report.append("")
    
    # 4. Automated Insights
    if insights:
        report.append("4. AUTOMATED ANALYTICAL INSIGHTS")
        report.append("-" * 55)
        for ins in insights:
            report.append(f"  [{ins.get('category', 'Insight').upper()}] {ins.get('title', '')}")
            report.append(f"  Observation : {ins.get('observation', '')}")
            report.append(f"  Methodology : {ins.get('methodology', '')}")
            report.append("")
            
    # 5. Dataset Comparison Summary (if provided)
    if comparison_summary and "delta_rows" in comparison_summary:
        report.append("5. DATASET COMPARISON SUMMARY")
        report.append("-" * 55)
        report.append(f"  Record Delta : {comparison_summary.get('rows_a')} -> {comparison_summary.get('rows_b')} (Change: {comparison_summary.get('delta_rows'):+d})")
        report.append(f"  Missing Cells: {comparison_summary.get('missing_cells_a')} -> {comparison_summary.get('missing_cells_b')}")
        report.append(f"  Duplicates   : {comparison_summary.get('duplicates_a')} -> {comparison_summary.get('duplicates_b')}")
        report.append("")
        
    # 6. Academic Methodology & Limitations
    report.append("=" * 78)
    report.append("METHODOLOGY & LIMITATIONS:")
    report.append("1. Preprocessing utilizes NumPy median imputation (np.nanmedian) and vectorized clipping.")
    report.append("2. Correlation uses Pearson's product-moment algorithm with zero-variance protection.")
    report.append("3. This software is engineered strictly for educational exploratory analysis.")
    report.append("=" * 78)
    
    return "\n".join(report)


def generate_html_summary_report(
    df: pd.DataFrame,
    audit_log: Dict[str, Any],
    stats_df: pd.DataFrame,
    corr_df: Optional[pd.DataFrame] = None,
    insights: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a print-ready, responsive HTML summary report suitable for browser viewing or saving as PDF.
    """
    timestamp = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
    stats_table_html = stats_df.to_html(index=False, classes="report-table", border=0) if not stats_df.empty else "<p>No statistics available.</p>"
    
    imputed_rows = ""
    for col, det in audit_log.get("imputed_details", {}).items():
        if isinstance(det, dict) and "imputed_count" in det:
            imputed_rows += f"<tr><td>{col}</td><td>{det.get('imputed_count', 0)}</td><td>{det.get('strategy', 'median').capitalize()}</td><td>{det.get('replacement_value', 'N/A')}</td></tr>"
    if not imputed_rows:
        imputed_rows = "<tr><td colspan='4'>No missing values required imputation.</td></tr>"
        
    clipped_rows = ""
    for col, count in audit_log.get("clipped_details", {}).items():
        clipped_rows += f"<tr><td>{col}</td><td>{count}</td><td>Biomedical bound clipping (np.clip)</td></tr>"
    if not clipped_rows:
        clipped_rows = "<tr><td colspan='3'>All records satisfied reference ranges.</td></tr>"

    insights_html = ""
    if insights:
        for ins in insights:
            insights_html += f"""
            <div style="background:#F8FAFC; border-left:4px solid #1E3A8A; padding:10px 14px; margin-bottom:10px; border-radius:4px;">
                <div style="font-size:11px; text-transform:uppercase; color:#64748B; font-weight:700;">{ins.get('category', 'Insight')}</div>
                <div style="font-weight:700; color:#1E3A8A; margin:2px 0;">{ins.get('title', '')}</div>
                <div style="font-size:13px; color:#334155;">{ins.get('observation', '')}</div>
                <div style="font-size:11px; color:#94A3B8; margin-top:4px;"><em>Rule: {ins.get('methodology', '')}</em></div>
            </div>
            """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Health Data Analysis & Visualization System Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1E293B;
        margin: 40px auto;
        max-width: 900px;
        padding: 0 20px;
    }}
    .header {{
        border-bottom: 3px solid #1E3A8A;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }}
    .header h1 {{
        color: #1E3A8A;
        margin: 0 0 8px 0;
        font-size: 26px;
    }}
    .header .subtitle {{
        color: #64748B;
        font-size: 14px;
        font-weight: 500;
    }}
    .badge {{
        display: inline-block;
        background-color: #0D9488;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 30px;
    }}
    .kpi-card {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }}
    .kpi-num {{
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
    }}
    .kpi-label {{
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
    }}
    h2 {{
        color: #0F172A;
        font-size: 18px;
        border-left: 4px solid #0D9488;
        padding-left: 10px;
        margin-top: 30px;
    }}
    .report-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 13px;
    }}
    .report-table th {{
        background-color: #F1F5F9;
        color: #334155;
        text-align: left;
        padding: 10px;
        border-bottom: 2px solid #CBD5E1;
    }}
    .report-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #E2E8F0;
    }}
    @media print {{
        body {{ margin: 15mm; }}
    }}
</style>
</head>
<body>

<div class="header">
    <span class="badge">MCA Minor Project</span>
    <h1>Health Data Analysis & Visualization System</h1>
    <div class="subtitle">Comprehensive Analytical Summary & Quality Audit | Generated: {timestamp}</div>
</div>

<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-num">{len(df)}</div>
        <div class="kpi-label">Active Records</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-num">{len(df.columns)}</div>
        <div class="kpi-label">Parameters</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-num">{audit_log.get('duplicates_removed', 0)}</div>
        <div class="kpi-label">Duplicates Purged</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-num">{sum(d.get('imputed_count', 0) for d in audit_log.get('imputed_details', {}).values() if isinstance(d, dict))}</div>
        <div class="kpi-label">Values Imputed</div>
    </div>
</div>

<h2>1. NumPy Data Cleaning & Preprocessing Audit</h2>
<p>Operations performed using NumPy vectorized operations and statistical medians:</p>
<table class="report-table">
    <thead>
        <tr><th>Target Parameter</th><th>Values Imputed</th><th>Strategy</th><th>Imputed Value</th></tr>
    </thead>
    <tbody>
        {imputed_rows}
    </tbody>
</table>

<table class="report-table" style="margin-top: 15px;">
    <thead>
        <tr><th>Parameter</th><th>Out-of-Range Count</th><th>Corrective Action</th></tr>
    </thead>
    <tbody>
        {clipped_rows}
    </tbody>
</table>

<h2>2. Statistical Profile of Health Indicators</h2>
{stats_table_html}

<h2>3. Automated Statistical Insights</h2>
{insights_html}

</body>
</html>
"""
    return html


def _sanitize_pdf_text(text: str) -> str:
    """Sanitizes text to safe ASCII for standard PDF Type-1 Helvetica font streams."""
    replacements = {
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2260": "!=",
        "\u00b1": "+/-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u00b2": "^2",
        "\u00b3": "^3"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    safe = text.encode('ascii', errors='replace').decode('ascii')
    return safe.replace("(", "\\(").replace(")", "\\)")


class SimplePDFWriter:
    """
    Pure Python standard PDF 1.4 generator.
    Produces valid, self-contained, printable PDF documents without any external libraries.
    """
    def __init__(self):
        self.objects = []
        self.pages = []
        self.streams = []

    def generate_pdf(self, title: str, subtitle: str, sections: List[Dict[str, Any]]) -> bytes:
        page_width = 612
        page_height = 792
        margin_x = 50
        
        pages_content = []
        current_lines = []
        
        def flush_page():
            if current_lines:
                pages_content.append(list(current_lines))
                current_lines.clear()
                
        current_lines.append(("HEADER", title, subtitle))
        y = 680
        
        for sec in sections:
            sec_title = sec.get("title", "")
            items = sec.get("items", [])
            
            if y < 140:
                flush_page()
                y = 720
                
            current_lines.append(("SECTION", sec_title))
            y -= 25
            
            for item in items:
                if y < 70:
                    flush_page()
                    y = 720
                current_lines.append(("TEXT", item))
                y -= 14
            y -= 10
            
        flush_page()
        
        num_pages = len(pages_content)
        page_obj_ids = []
        stream_obj_ids = []
        next_id = 5
        
        for _ in range(num_pages):
            page_obj_ids.append(next_id)
            stream_obj_ids.append(next_id + 1)
            next_id += 2
            
        catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
        kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
        pages_obj = f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>".encode('ascii')
        font1 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        font2 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        
        objects_dict = {1: catalog, 2: pages_obj, 3: font1, 4: font2}
        
        for p_idx, p_data in enumerate(pages_content):
            stream_cmds = []
            cur_y = 730
            
            stream_cmds.append("0.117 0.227 0.541 rg")  # #1E3A8A
            stream_cmds.append(f"{margin_x} 750 {page_width - 2 * margin_x} 3 re f")
            
            for entry in p_data:
                etype = entry[0]
                if etype == "HEADER":
                    t, sub = entry[1], entry[2]
                    stream_cmds.append("BT")
                    stream_cmds.append("/F2 16 Tf")
                    stream_cmds.append("0.117 0.227 0.541 rg")
                    stream_cmds.append(f"{margin_x} 720 Td")
                    stream_cmds.append(f"({_sanitize_pdf_text(t)}) Tj")
                    stream_cmds.append("ET")
                    
                    stream_cmds.append("BT")
                    stream_cmds.append("/F1 9 Tf")
                    stream_cmds.append("0.392 0.455 0.545 rg")
                    stream_cmds.append(f"{margin_x} 704 Td")
                    stream_cmds.append(f"({_sanitize_pdf_text(sub)}) Tj")
                    stream_cmds.append("ET")
                    cur_y = 675
                    
                elif etype == "SECTION":
                    stream_cmds.append("BT")
                    stream_cmds.append("/F2 12 Tf")
                    stream_cmds.append("0.051 0.580 0.533 rg")
                    stream_cmds.append(f"{margin_x} {cur_y} Td")
                    stream_cmds.append(f"({_sanitize_pdf_text(entry[1])}) Tj")
                    stream_cmds.append("ET")
                    cur_y -= 18
                    
                elif etype == "TEXT":
                    stream_cmds.append("BT")
                    stream_cmds.append("/F1 9.5 Tf")
                    stream_cmds.append("0.12 0.16 0.23 rg")
                    stream_cmds.append(f"{margin_x} {cur_y} Td")
                    stream_cmds.append(f"({_sanitize_pdf_text(entry[1])}) Tj")
                    stream_cmds.append("ET")
                    cur_y -= 13
                    
            stream_cmds.append("BT")
            stream_cmds.append("/F1 8 Tf")
            stream_cmds.append("0.5 0.5 0.5 rg")
            stream_cmds.append(f"{margin_x} 30 Td")
            foot_msg = f"Health Data Analysis System | Page {p_idx + 1} of {num_pages}"
            stream_cmds.append(f"({foot_msg}) Tj")
            stream_cmds.append("ET")
            
            stream_bytes = "\n".join(stream_cmds).encode('latin-1')
            stream_def = f"<< /Length {len(stream_bytes)} >>\nstream\n".encode('ascii') + stream_bytes + b"\nendstream"
            
            p_obj_id = page_obj_ids[p_idx]
            s_obj_id = stream_obj_ids[p_idx]
            p_def = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
                f"/Contents {s_obj_id} 0 R /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
            ).encode('ascii')
            
            objects_dict[p_obj_id] = p_def
            objects_dict[s_obj_id] = stream_def
            
        pdf_out = io.BytesIO()
        pdf_out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        
        xref_offsets = {}
        sorted_ids = sorted(objects_dict.keys())
        
        for obj_id in sorted_ids:
            xref_offsets[obj_id] = pdf_out.tell()
            pdf_out.write(f"{obj_id} 0 obj\n".encode('ascii'))
            pdf_out.write(objects_dict[obj_id])
            pdf_out.write(b"\nendobj\n")
            
        xref_start = pdf_out.tell()
        pdf_out.write(b"xref\n")
        pdf_out.write(f"0 {len(sorted_ids) + 1}\n".encode('ascii'))
        pdf_out.write(b"0000000000 65535 f \n")
        
        for obj_id in sorted_ids:
            offset = xref_offsets[obj_id]
            pdf_out.write(f"{offset:010d} 00000 n \n".encode('ascii'))
            
        pdf_out.write(b"trailer\n")
        pdf_out.write(f"<< /Size {len(sorted_ids) + 1} /Root 1 0 R >>\n".encode('ascii'))
        pdf_out.write(b"startxref\n")
        pdf_out.write(f"{xref_start}\n".encode('ascii'))
        pdf_out.write(b"%%EOF")
        
        return pdf_out.getvalue()


def generate_pdf_report(
    df: pd.DataFrame,
    audit_log: Dict[str, Any],
    stats_df: pd.DataFrame,
    corr_df: Optional[pd.DataFrame] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    comparison_summary: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Constructs and returns binary PDF bytes containing all required analytical findings.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    writer = SimplePDFWriter()
    sections = []
    
    # 1. Overview
    overview_items = [
        f"Report Timestamp       : {timestamp}",
        f"Total Patients Analyzed: {len(df)}",
        f"Total Attributes       : {len(df.columns)}",
        f"Engine Architecture    : Python 3.13 + NumPy Vectorization"
    ]
    sections.append({"title": "1. Dataset Executive Overview", "items": overview_items})
    
    # 2. Cleaning Audit
    cleaning_items = [
        f"Initial Records Imported       : {audit_log.get('initial_rows', len(df))}",
        f"Duplicate Patient Records Purged: {audit_log.get('duplicates_removed', 0)}",
        f"Imputation Method Applied       : {audit_log.get('imputation_strategy', 'median').capitalize()} Imputation"
    ]
    for col, det in audit_log.get("imputed_details", {}).items():
        if isinstance(det, dict) and "imputed_count" in det:
            cleaning_items.append(f"  - {col}: {det['imputed_count']} values imputed ({det.get('replacement_value')})")
    for col, count in audit_log.get("clipped_details", {}).items():
        cleaning_items.append(f"  - {col}: {count} outlier values bounded via np.clip()")
    cleaning_items.append(f"BMI Recalculations: {audit_log.get('bmi_recalculated_count', 0)} rows updated via vectorized formula")
    sections.append({"title": "2. Data Quality & Preprocessing Audit", "items": cleaning_items})
    
    # 3. Descriptive Stats
    stats_items = []
    if not stats_df.empty:
        stats_items.append(f"{'Parameter':<16} | {'Mean':>7} | {'Median':>7} | {'StdDev':>7} | {'Min':>6} | {'Max':>6}")
        stats_items.append("-" * 62)
        for _, r in stats_df.iterrows():
            stats_items.append(
                f"{str(r['Variable']):<16} | {float(r['Mean']):>7.1f} | {float(r['Median']):>7.1f} | "
                f"{float(r['Std_Dev']):>7.1f} | {float(r['Min']):>6.1f} | {float(r['Max']):>6.1f}"
            )
    sections.append({"title": "3. Descriptive Statistics Summary", "items": stats_items})
    
    # 4. Correlation
    if corr_df is not None and not corr_df.empty:
        corr_items = []
        cols = corr_df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_df.iloc[i, j]
                if abs(val) >= 0.35:
                    corr_items.append(f"  * {cols[i]} <-> {cols[j]}: r = {val:+.3f}")
        if not corr_items:
            corr_items.append("No linear correlations exceeded |r| >= 0.35.")
        corr_items.append("Note: Statistical correlation indicates association, not medical causality.")
        sections.append({"title": "4. Key Biomarker Correlations", "items": corr_items})
        
    # 5. Automated Insights
    if insights:
        ins_items = []
        for ins in insights[:5]:
            ins_items.append(f"[{ins.get('category')}] {ins.get('title')}: {ins.get('observation')}")
        sections.append({"title": "5. Automated Statistical Insights", "items": ins_items})
        
    # 6. Comparison (if provided)
    if comparison_summary and "delta_rows" in comparison_summary:
        comp_items = [
            f"Compared Cohorts : {comparison_summary.get('label_a')} vs {comparison_summary.get('label_b')}",
            f"Record Count Delta: {comparison_summary.get('rows_a')} -> {comparison_summary.get('rows_b')} ({comparison_summary.get('delta_rows'):+d})",
            f"Missing Cell Delta: {comparison_summary.get('missing_cells_a')} -> {comparison_summary.get('missing_cells_b')}"
        ]
        sections.append({"title": "6. Dataset Comparison Summary", "items": comp_items})
        
    # 7. Methodology & Limitations
    disc_items = [
        "1. All analytical estimations are conducted on valid data points using standard statistical formulas.",
        "2. Correlation indicates mathematical association and does not establish medical causation.",
        "3. System is built for educational exploratory data analysis and student demonstration."
    ]
    sections.append({"title": "7. Methodology & Analytical Limitations", "items": disc_items})
    
    return writer.generate_pdf(
        title="Health Data Analysis & Visualization System",
        subtitle="Analytical & Statistical Summary Report",
        sections=sections
    )
