"""
=============================================================================
Health Data Analysis and Visualization System Using Python
=============================================================================
Main Application File: app.py
Framework: Streamlit
Technologies: Python 3, NumPy, Pandas, Matplotlib, Seaborn, Streamlit
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

# Import modular project components
from modules.data_loader import (
    load_csv_data,
    load_default_sample_data,
    get_dataset_metadata,
    validate_health_columns,
    NUMERIC_COLUMNS
)
from modules.data_cleaning import (
    detect_missing_values,
    detect_duplicates,
    detect_out_of_range,
    detect_iqr_outliers,
    detect_empty_and_constant_columns,
    perform_full_cleaning_pipeline
)
from modules.statistical_analysis import (
    generate_dataset_summary_statistics,
    compute_correlation_matrix_numpy,
    interpret_correlation_strength,
    stratify_clinical_risks,
    DEFAULT_AGE_BINS,
    DEFAULT_AGE_LABELS
)
from modules.visualization import (
    plot_age_distribution,
    plot_gender_distribution,
    plot_age_group_distribution,
    plot_bmi_distribution,
    plot_demographics_overview,
    plot_vital_signs_boxplots,
    plot_vital_distributions,
    plot_correlation_heatmap,
    plot_scatter_relationship,
    plot_clinical_risk_charts
)
from modules.insights import generate_automated_insights
from modules.dataset_comparison import compare_datasets, plot_dataset_comparison
from modules.history_manager import (
    save_history_entry,
    load_history_entries,
    delete_history_entry,
    clear_all_history,
    export_history_json
)
from modules.report_generator import (
    generate_text_summary_report,
    generate_html_summary_report,
    generate_pdf_report
)

# ---------------------------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Data Analysis System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Dashboard Styling
st.markdown("""
<style>
    /* Top Header Styling */
    .app-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%);
        color: white;
        padding: 22px 26px;
        border-radius: 10px;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .app-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: white;
    }
    .app-header p {
        margin: 4px 0 0 0;
        font-size: 13.5px;
        opacity: 0.92;
    }
    
    /* Modern Metric KPI Cards */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: transform 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-lbl {
        font-size: 11.5px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }
    
    /* Insight Cards */
    .insight-box {
        background-color: #F8FAFC;
        border-left: 4px solid #1E3A8A;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .insight-cat {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #0D9488;
    }
    .insight-title {
        font-size: 14px;
        font-weight: 700;
        color: #0F172A;
        margin: 2px 0 4px 0;
    }
    .insight-desc {
        font-size: 13px;
        color: #334155;
    }
    .insight-meta {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 4px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    initial_df, _ = load_default_sample_data()
    st.session_state.raw_df = initial_df
    st.session_state.data_source = "Synthetic Benchmark (163 records)"

if "cleaned_df" not in st.session_state:
    if st.session_state.raw_df is not None:
        c_df, audit = perform_full_cleaning_pipeline(st.session_state.raw_df, imputation_strategy="median")
        st.session_state.cleaned_df = c_df
        st.session_state.cleaning_audit = audit
    else:
        st.session_state.cleaned_df = None
        st.session_state.cleaning_audit = {}


# ---------------------------------------------------------------------------
# Sidebar: Navigation & Global Filters
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Health Analysis System")
st.sidebar.caption("Interactive Data Analytics & Visualization Platform")

# 9 Cohesive Navigation Sections (Notice: About project is removed as requested)
SECTIONS = [
    "📊 1. Overview Dashboard",
    "🔍 2. Data Explorer & Upload",
    "🧹 3. Data Cleaning & Quality",
    "🔗 4. Correlation Explorer",
    "💡 5. Automated Insights",
    "⚖️ 6. Dataset Comparison",
    "📋 7. Patient Records",
    "📑 8. Export & Reports",
    "🕒 9. Analysis History"
]

selected_section = st.sidebar.radio("Navigation:", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Active Cohort Filters")

use_cleaned = st.sidebar.checkbox("Use Cleaned Dataset", value=True, help="Toggle between raw dataset and NumPy-cleaned dataset.")
current_df = st.session_state.cleaned_df if (use_cleaned and st.session_state.cleaned_df is not None) else st.session_state.raw_df

filtered_df = current_df.copy() if current_df is not None else pd.DataFrame()

# Build dynamic filters based on available columns
if not filtered_df.empty:
    if "Gender" in filtered_df.columns:
        available_genders = filtered_df["Gender"].dropna().unique().tolist()
        if available_genders:
            selected_genders = st.sidebar.multiselect("Gender:", options=available_genders, default=available_genders)
            if selected_genders:
                filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]
            
    if "Age" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["Age"]):
        valid_ages = filtered_df["Age"].dropna()
        if not valid_ages.empty:
            min_age, max_age = int(valid_ages.min()), int(valid_ages.max())
            if min_age < max_age:
                age_range = st.sidebar.slider("Age (Years):", min_value=min_age, max_value=max_age, value=(min_age, max_age))
                filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

    if "Systolic_BP" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["Systolic_BP"]):
        valid_sys = filtered_df["Systolic_BP"].dropna()
        if not valid_sys.empty:
            min_sys, max_sys = int(valid_sys.min()), int(valid_sys.max())
            if min_sys < max_sys:
                sys_range = st.sidebar.slider("Systolic BP (mmHg):", min_value=min_sys, max_value=max_sys, value=(min_sys, max_sys))
                filtered_df = filtered_df[(filtered_df["Systolic_BP"] >= sys_range[0]) & (filtered_df["Systolic_BP"] <= sys_range[1])]

    if "Blood_Glucose" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["Blood_Glucose"]):
        valid_glu = filtered_df["Blood_Glucose"].dropna()
        if not valid_glu.empty:
            min_glu, max_glu = float(valid_glu.min()), float(valid_glu.max())
            if min_glu < max_glu:
                glu_range = st.sidebar.slider("Blood Glucose (mg/dL):", min_value=round(min_glu), max_value=round(max_glu), value=(round(min_glu), round(max_glu)))
                filtered_df = filtered_df[(filtered_df["Blood_Glucose"] >= glu_range[0]) & (filtered_df["Blood_Glucose"] <= glu_range[1])]

    st.sidebar.markdown(f"**Cohort Count:** {len(filtered_df)} / {len(current_df)} records")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()
with col_btn2:
    if st.button("🗑️ Reset All", use_container_width=True, help="Reloads sample data and clears all session state."):
        def_df, _ = load_default_sample_data()
        st.session_state.raw_df = def_df
        st.session_state.data_source = "Synthetic Benchmark (163 records)"
        c_df, audit = perform_full_cleaning_pipeline(def_df)
        st.session_state.cleaned_df = c_df
        st.session_state.cleaning_audit = audit
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data Mode: " + ("Cleaned (NumPy)" if use_cleaned else "Raw Dataset"))
st.sidebar.caption("Source: " + str(st.session_state.get("data_source", "Unknown")))


# ---------------------------------------------------------------------------
# Global Header (Disclaimer banner permanently removed as requested)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>Health Data Analysis and Visualization System</h1>
    <p>Advanced Exploratory Data Analysis, NumPy Data Quality Cleaning, Correlation Engine & Automated Insights</p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# SECTION 1: OVERVIEW DASHBOARD
# ===========================================================================
if selected_section.startswith("📊 1"):
    st.subheader("📊 Executive Health Metrics Dashboard")
    
    if filtered_df.empty:
        st.warning("No records match the active filter criteria. Please adjust your sidebar filters.")
    else:
        # Dynamic Top KPI Summary Cards (detects available columns dynamically)
        total_rec = len(filtered_df)
        num_cols = len([c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])])
        missing_cnt = int(filtered_df.isna().sum().sum())
        dup_cnt = int(filtered_df.duplicated().sum())
        iqr_df = detect_iqr_outliers(filtered_df)
        outlier_cnt = int(iqr_df["Outlier Count"].sum()) if not iqr_df.empty else 0
        avg_bmi = float(filtered_df["BMI"].mean()) if "BMI" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["BMI"]) else None

        kpi_cols = st.columns(6)
        kpi_cols[0].markdown(f'<div class="metric-card"><div class="metric-val">{total_rec:,}</div><div class="metric-lbl">Total Records</div></div>', unsafe_allow_html=True)
        kpi_cols[1].markdown(f'<div class="metric-card"><div class="metric-val">{num_cols}</div><div class="metric-lbl">Numeric Fields</div></div>', unsafe_allow_html=True)
        kpi_cols[2].markdown(f'<div class="metric-card"><div class="metric-val">{missing_cnt}</div><div class="metric-lbl">Missing Cells</div></div>', unsafe_allow_html=True)
        kpi_cols[3].markdown(f'<div class="metric-card"><div class="metric-val">{dup_cnt}</div><div class="metric-lbl">Duplicate Rows</div></div>', unsafe_allow_html=True)
        kpi_cols[4].markdown(f'<div class="metric-card"><div class="metric-val">{outlier_cnt}</div><div class="metric-lbl">IQR Outliers</div></div>', unsafe_allow_html=True)
        bmi_display = f"{avg_bmi:.1f}" if avg_bmi is not None else "N/A"
        kpi_cols[5].markdown(f'<div class="metric-card"><div class="metric-val">{bmi_display}</div><div class="metric-lbl">Avg. BMI</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clinical Risk Stratification Overview (if clinical vitals are available)
        risks = stratify_clinical_risks(filtered_df)
        if risks:
            st.markdown("### 🏥 Clinical Category Distributions (AHA & ADA Standards)")
            fig_risks = plot_clinical_risk_charts(filtered_df)
            st.pyplot(fig_risks)
            st.markdown("---")
            
        # Two-column high-level distributions
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 👥 Age Distribution")
            if "Age" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["Age"]):
                fig_age = plot_age_distribution(filtered_df)
                st.pyplot(fig_age)
            else:
                st.info("Age column unavailable in the loaded dataset.")
                
        with col_right:
            st.markdown("#### ⚖️ BMI Classification")
            if "BMI" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["BMI"]):
                fig_bmi = plot_bmi_distribution(filtered_df)
                st.pyplot(fig_bmi)
            else:
                st.info("BMI column unavailable in the loaded dataset.")


# ===========================================================================
# SECTION 2: DATA EXPLORER & UPLOAD
# ===========================================================================
elif selected_section.startswith("🔍 2"):
    st.subheader("🔍 Dataset Ingestion & Schema Inspection")
    st.write("Upload custom patient CSV datasets or reload the built-in synthetic benchmark.")
    
    upload_tab1, upload_tab2 = st.tabs(["📤 Upload Custom CSV", "📦 Built-in Benchmark Data"])
    
    with upload_tab1:
        uploaded_file = st.file_uploader(
            "Upload CSV File (Compatible fields: Age, Gender, Height, Weight, BP, Glucose, etc.):",
            type=["csv"],
            help="Files are processed entirely in local memory. No data is stored externally."
        )
        if uploaded_file is not None:
            new_df, error = load_csv_data(uploaded_file)
            if error:
                st.error(f"Error loading uploaded dataset: {error}")
            else:
                st.success(f"File '{uploaded_file.name}' loaded successfully! ({len(new_df)} records)")
                st.session_state.raw_df = new_df
                st.session_state.data_source = f"Uploaded: {uploaded_file.name}"
                c_df, audit = perform_full_cleaning_pipeline(new_df)
                st.session_state.cleaned_df = c_df
                st.session_state.cleaning_audit = audit
                st.rerun()
                
    with upload_tab2:
        st.write("The built-in benchmark dataset contains 163 patient records with realistic physiological distributions and intentional missing, duplicate, and out-of-range observations for demonstration.")
        if st.button("🔄 Reload Default Benchmark Dataset"):
            def_df, _ = load_default_sample_data()
            st.session_state.raw_df = def_df
            st.session_state.data_source = "Synthetic Benchmark (163 records)"
            c_df, audit = perform_full_cleaning_pipeline(def_df)
            st.session_state.cleaned_df = c_df
            st.session_state.cleaning_audit = audit
            st.success("Default benchmark dataset reloaded!")
            st.rerun()
            
    st.markdown("---")
    
    if st.session_state.raw_df is not None:
        meta = get_dataset_metadata(st.session_state.raw_df)
        val = validate_health_columns(st.session_state.raw_df)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Records", meta["total_records"])
        m_col2.metric("Total Columns", meta["total_columns"])
        m_col3.metric("Numeric Fields", meta["numeric_columns_count"])
        m_col4.metric("Standard Schema Coverage", f"{val['coverage_percentage']}%")
        
        st.markdown("#### 📋 Column Schema & Missing Values")
        st.dataframe(meta["column_info"], use_container_width=True)
        
        st.markdown("#### 👀 Dataset Preview")
        preview_limit = st.slider("Preview Rows:", 5, min(50, len(st.session_state.raw_df)), 10)
        st.dataframe(st.session_state.raw_df.head(preview_limit), use_container_width=True)


# ===========================================================================
# SECTION 3: DATA CLEANING & QUALITY
# ===========================================================================
elif selected_section.startswith("🧹 3"):
    st.subheader("🧹 Smart Data Quality Analyzer & NumPy Cleaning Engine")
    st.write("Comprehensive data quality diagnosis, IQR statistical outlier detection, and vectorized NumPy cleaning.")
    
    if st.session_state.raw_df is None:
        st.warning("Please upload or load a dataset first.")
    else:
        # Step A: Diagnostics
        d_tab1, d_tab2, d_tab3 = st.tabs(["🔍 Missing Values & Duplicates", "📊 Statistical Outliers (IQR)", "⚠️ Constant & Empty Columns"])
        
        with d_tab1:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("##### Missing Values Detection (`pd.api.types` & `np.isnan`)")
                missing_report = detect_missing_values(st.session_state.raw_df)
                st.dataframe(missing_report, use_container_width=True)
            with col_m2:
                st.markdown("##### Duplicate Records Inspection")
                dup_info = detect_duplicates(st.session_state.raw_df)
                st.write(f"• **Exact Duplicate Rows:** `{dup_info['full_duplicates_count']}`")
                st.write(f"• **Duplicate Patient IDs:** `{dup_info['pid_duplicates_count']}`")
                if dup_info["has_duplicates"]:
                    st.warning("Duplicate patient entries detected. Recommend purging duplicates in the pipeline below.")
                else:
                    st.success("No duplicate records detected.")
                    
        with d_tab2:
            st.markdown("##### Statistical Outlier Detection (Interquartile Range Method)")
            st.caption("Outlier criterion: Observations below $Q_1 - 1.5 \\times \\text{IQR}$ or above $Q_3 + 1.5 \\times \\text{IQR}$.")
            iqr_outliers = detect_iqr_outliers(st.session_state.raw_df)
            if not iqr_outliers.empty:
                st.dataframe(iqr_outliers, use_container_width=True)
            else:
                st.success("No statistical IQR outliers detected in numeric fields.")
                
            physio_anomalies = detect_out_of_range(st.session_state.raw_df)
            if not physio_anomalies.empty:
                st.markdown("##### Physiological Boundary Violations")
                st.dataframe(physio_anomalies, use_container_width=True)
                
        with d_tab3:
            st.markdown("##### Structural Column Validation")
            empty_const = detect_empty_and_constant_columns(st.session_state.raw_df)
            st.write(f"• **Empty Columns (100% Null):** `{empty_const['empty_columns'] or 'None'}`")
            st.write(f"• **Constant Columns (Zero Variance):** `{empty_const['constant_columns'] or 'None'}`")
            
        st.markdown("---")
        
        # Step B: Cleaning Engine Controls
        st.markdown("### ⚙️ Configure Cleaning Pipeline")
        st.caption("Cleaning is always performed on an independent working copy. The original uploaded dataset is never overwritten.")
        
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            impute_strategy = st.selectbox(
                "Missing Value Strategy:",
                ["median", "mean", "mode", "drop"],
                help="Median imputation is resilient against clinical outliers."
            )
        with c_col2:
            purge_dups = st.checkbox("Purge Duplicate Rows", value=True)
        with c_col3:
            clip_bounds = st.checkbox("Bound Physiological Outliers (np.clip)", value=True)
        with c_col4:
            treat_iqr = st.checkbox("Treat IQR Outliers (Winsorize)", value=False)
            
        if st.button("🚀 Execute Data Cleaning Pipeline", type="primary"):
            cleaned, audit = perform_full_cleaning_pipeline(
                st.session_state.raw_df,
                imputation_strategy=impute_strategy,
                clip_out_of_range=clip_bounds,
                handle_iqr=treat_iqr,
                iqr_strategy="clip"
            )
            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_audit = audit
            st.success("Data Cleaning Pipeline Executed Successfully!")
            st.rerun()
            
        # Step C: Audit Log
        if st.session_state.cleaning_audit:
            st.markdown("### 📜 Cleaning Audit Report & Impact")
            audit = st.session_state.cleaning_audit
            
            a_col1, a_col2, a_col3, a_col4 = st.columns(4)
            a_col1.metric("Rows Before", audit.get("initial_rows", 0))
            a_col2.metric("Duplicates Removed", audit.get("duplicates_removed", 0))
            a_col3.metric("Rows After", audit.get("final_rows", 0))
            a_col4.metric("BMI Recalculations", audit.get("bmi_recalculated_count", 0))
            
            imputed_map = audit.get("imputed_details", {})
            if imputed_map:
                st.markdown("**Imputed Columns:**")
                imp_summary = [
                    {"Parameter": col, "Values Replaced": d.get("imputed_count", 0), "Strategy": d.get("strategy", "").capitalize(), "Replacement": d.get("replacement_value")}
                    for col, d in imputed_map.items() if isinstance(d, dict)
                ]
                st.dataframe(pd.DataFrame(imp_summary), use_container_width=True)
                
            st.markdown("#### 🔬 Cleaned Dataset Sample")
            st.dataframe(st.session_state.cleaned_df.head(8), use_container_width=True)


# ===========================================================================
# SECTION 4: CORRELATION EXPLORER
# ===========================================================================
elif selected_section.startswith("🔗 4"):
    st.subheader("🔗 Interactive Correlation & Bivariate Explorer")
    st.write("Explore linear (Pearson) and monotonic (Spearman) relationships between numerical variables.")
    
    if filtered_df.empty:
        st.warning("No records available to compute correlation under active filters.")
    else:
        num_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])]
        
        if len(num_cols) < 2:
            st.info("At least two numerical variables are required to perform correlation analysis.")
        else:
            corr_tab1, corr_tab2 = st.tabs(["🔥 Correlation Heatmap", "🔍 Bivariate Scatter Explorer"])
            
            with corr_tab1:
                method_choice = st.radio("Correlation Method:", ["pearson", "spearman"], horizontal=True,
                                         help="Pearson evaluates linear relationships; Spearman evaluates monotonic rank relationships.")
                
                corr_matrix, warnings = compute_correlation_matrix_numpy(filtered_df, columns=num_cols, method=method_choice)
                if warnings:
                    for w in warnings:
                        st.info(w)
                        
                fig_corr = plot_correlation_heatmap(corr_matrix, method_name=f"{method_choice.capitalize()} r")
                st.pyplot(fig_corr)
                
            with corr_tab2:
                sc_col1, sc_col2, sc_col3 = st.columns(3)
                
                def_x_idx = num_cols.index("Age") if "Age" in num_cols else 0
                def_y_idx = num_cols.index("Systolic_BP") if "Systolic_BP" in num_cols else min(1, len(num_cols)-1)
                
                with sc_col1:
                    x_var = st.selectbox("X-Axis Variable:", num_cols, index=def_x_idx)
                with sc_col2:
                    y_var = st.selectbox("Y-Axis Variable:", num_cols, index=def_y_idx)
                with sc_col3:
                    hue_var = st.selectbox("Color By (Hue):", ["Gender", "None"] if "Gender" in filtered_df.columns else ["None"], index=0)
                    hue_choice = None if hue_var == "None" else "Gender"
                    
                fig_scatter, r_val, n_obs = plot_scatter_relationship(filtered_df, x_var, y_var, hue_col=hue_choice, method=method_choice)
                st.pyplot(fig_scatter)
                
                interp = interpret_correlation_strength(r_val, method_name=f"{method_choice.capitalize()} r")
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-cat">Statistical Interpretation</div>
                    <div class="insight-title">{x_var} vs {y_var} (N = {n_obs})</div>
                    <div class="insight-desc">
                        • <strong>Correlation Coefficient:</strong> {interp['r_value']}<br>
                        • <strong>Strength & Direction:</strong> {interp['strength']} {interp['direction']}<br>
                        • <strong>Explanation:</strong> {interp['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ===========================================================================
# SECTION 5: AUTOMATED INSIGHTS
# ===========================================================================
elif selected_section.startswith("💡 5"):
    st.subheader("💡 Rule-Based Automated Insights Generator")
    st.write("Generates actionable findings derived from actual dataset statistics and physiological standards.")
    
    if filtered_df.empty:
        st.warning("No records available to compute insights.")
    else:
        insights = generate_automated_insights(
            filtered_df,
            audit_log=st.session_state.get("cleaning_audit"),
            raw_df=st.session_state.get("raw_df")
        )
        
        for ins in insights:
            badge_color = "#1E3A8A"
            if ins.get("badge") == "warning":
                badge_color = "#DC2626"
            elif ins.get("badge") == "success":
                badge_color = "#059669"
            elif ins.get("badge") == "info":
                badge_color = "#0D9488"
                
            st.markdown(f"""
            <div style="background:#F8FAFC; border-left:4px solid {badge_color}; padding:14px 18px; border-radius:6px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                <div style="font-size:11px; text-transform:uppercase; color:{badge_color}; font-weight:700;">{ins.get('category')}</div>
                <div style="font-size:15px; font-weight:700; color:#0F172A; margin:2px 0 6px 0;">{ins.get('title')}</div>
                <div style="font-size:13.5px; color:#334155; line-height:1.5;">{ins.get('observation')}</div>
                <div style="font-size:11.5px; color:#94A3B8; margin-top:6px;"><em>Statistical Basis: {ins.get('methodology')}</em></div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# SECTION 6: DATASET COMPARISON
# ===========================================================================
elif selected_section.startswith("⚖️ 6"):
    st.subheader("⚖️ Dataset Comparison Module")
    st.write("Compare the Raw vs. Cleaned datasets, or upload a secondary dataset for side-by-side analysis.")
    
    comp_mode = st.radio("Comparison Mode:", ["Raw Dataset vs. Cleaned Dataset", "Upload Secondary Dataset to Compare"], horizontal=True)
    
    df_compare_target = None
    label_target = ""
    
    if comp_mode == "Raw Dataset vs. Cleaned Dataset":
        if st.session_state.raw_df is not None and st.session_state.cleaned_df is not None:
            comp_result = compare_datasets(
                st.session_state.raw_df,
                st.session_state.cleaned_df,
                label_a="Raw Dataset",
                label_b="Cleaned Dataset"
            )
        else:
            comp_result = {"error": "Raw or cleaned dataset is unavailable."}
    else:
        sec_file = st.file_uploader("Upload Secondary CSV for Comparison:", type=["csv"], key="sec_compare_csv")
        if sec_file is not None:
            sec_df, sec_err = load_csv_data(sec_file)
            if sec_err:
                st.error(f"Error loading secondary dataset: {sec_err}")
                comp_result = {"error": sec_err}
            else:
                comp_result = compare_datasets(
                    st.session_state.raw_df,
                    sec_df,
                    label_a="Primary Dataset",
                    label_b=f"Uploaded ({sec_file.name})"
                )
        else:
            st.info("Upload a secondary CSV file above to compare with the active primary dataset.")
            comp_result = None

    if comp_result and "error" not in comp_result:
        # Comparison Metrics
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.metric("Rows A vs B", f"{comp_result['rows_a']} vs {comp_result['rows_b']}", delta=comp_result['delta_rows'])
        c_kpi2.metric("Columns A vs B", f"{comp_result['cols_a']} vs {comp_result['cols_b']}")
        c_kpi3.metric("Missing Cells A vs B", f"{comp_result['missing_cells_a']} vs {comp_result['missing_cells_b']}", delta=comp_result['missing_cells_b'] - comp_result['missing_cells_a'])
        c_kpi4.metric("Duplicates A vs B", f"{comp_result['duplicates_a']} vs {comp_result['duplicates_b']}")
        
        st.markdown("#### 📊 Comparative Charts")
        fig_comp = plot_dataset_comparison(comp_result)
        st.pyplot(fig_comp)
        
        st.markdown("#### 📐 Statistical Drift Table (Shared Numeric Attributes)")
        st.dataframe(comp_result["stats_comparison"], use_container_width=True)
        
        st.markdown("#### 🔍 Missingness Delta by Column")
        st.dataframe(comp_result["missing_col_comparison"], use_container_width=True)


# ===========================================================================
# SECTION 7: PATIENT RECORDS
# ===========================================================================
elif selected_section.startswith("📋 7"):
    st.subheader("📋 Patient Records Browser & Filtered Search")
    st.write("Search, paginate, sort, and inspect patient records matching active filters.")
    
    if filtered_df.empty:
        st.warning("No records match the active filter criteria.")
    else:
        search_query = st.text_input("🔎 Search by Patient ID:", placeholder="e.g. PID-1015")
        display_df = filtered_df.copy()
        
        if search_query and "Patient_ID" in display_df.columns:
            display_df = display_df[display_df["Patient_ID"].astype(str).str.contains(search_query, case=False, na=False)]
            
        st.caption(f"Showing **{len(display_df)}** records:")
        
        page_size = st.selectbox("Records per page:", [10, 25, 50, 100], index=0)
        total_pages = max(1, (len(display_df) - 1) // page_size + 1)
        current_page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)
        st.caption(f"Page {current_page} of {total_pages}")
        
        csv_buffer = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Cohort (.csv)",
            data=csv_buffer,
            file_name="filtered_health_records.csv",
            mime="text/csv"
        )


# ===========================================================================
# SECTION 8: EXPORT & REPORTS
# ===========================================================================
elif selected_section.startswith("📑 8"):
    st.subheader("📑 Report Generation & Multi-Format Data Export")
    st.write("Generate one-click PDF analysis reports, cleaned CSV datasets, and print-ready HTML medical summaries.")
    
    if current_df is None or current_df.empty:
        st.warning("No dataset loaded.")
    else:
        num_cols = [c for c in current_df.columns if pd.api.types.is_numeric_dtype(current_df[c])]
        summary_stats = generate_dataset_summary_statistics(current_df, target_columns=num_cols)
        corr_m, _ = compute_correlation_matrix_numpy(current_df, columns=num_cols)
        audit_log = st.session_state.get("cleaning_audit", {})
        insights_list = generate_automated_insights(current_df, audit_log=audit_log)
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            st.markdown("#### 📑 Standalone PDF Report")
            st.write("Generates a multi-page binary PDF report containing dataset metrics, cleaning audit, descriptive statistics, and automated insights.")
            pdf_bytes = generate_pdf_report(current_df, audit_log, summary_stats, corr_m, insights=insights_list)
            st.download_button(
                label="📑 Download Full PDF Report (.pdf)",
                data=pdf_bytes,
                file_name="health_analysis_report.pdf",
                mime="application/pdf",
                type="primary"
            )
            
        with exp_col2:
            st.markdown("#### 💾 Cleaned Dataset (CSV)")
            st.write("Download the processed dataset with duplicates purged, missing values imputed, and BMI calculated.")
            cleaned_csv = current_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV (.csv)",
                data=cleaned_csv,
                file_name="cleaned_health_data.csv",
                mime="text/csv"
            )
            
        with exp_col3:
            st.markdown("#### 📊 Statistical Summary (CSV)")
            st.write("Download the complete 11-point descriptive statistics table across all numeric variables.")
            stats_csv = summary_stats.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Statistics (.csv)",
                data=stats_csv,
                file_name="health_statistics_summary.csv",
                mime="text/csv"
            )
            
        st.markdown("---")
        
        html_rep = generate_html_summary_report(current_df, audit_log, summary_stats, corr_m, insights=insights_list)
        st.download_button(
            label="📄 Download Print-Ready HTML Medical Audit Report (.html)",
            data=html_rep.encode('utf-8'),
            file_name="health_medical_audit_report.html",
            mime="text/html"
        )
        
        st.markdown("#### 📄 On-Screen Analytical Report Preview")
        text_rep = generate_text_summary_report(current_df, audit_log, summary_stats, corr_m, insights=insights_list)
        st.text_area("Summary Report Output:", text_rep, height=300)


# ===========================================================================
# SECTION 9: ANALYSIS HISTORY
# ===========================================================================
elif selected_section.startswith("🕒 9"):
    st.subheader("🕒 Local Analysis History Manager")
    st.write("Save, inspect, and export lightweight metadata snapshots of your previous analysis runs.")
    st.caption("Saved history stores summary metrics and logs locally; no raw patient data is stored.")
    
    hist_col1, hist_col2 = st.columns([2, 1])
    
    with hist_col1:
        if st.button("💾 Save Current Analysis Snapshot to History", type="primary"):
            audit_log = st.session_state.get("cleaning_audit", {})
            insights_list = generate_automated_insights(current_df, audit_log=audit_log)
            top_ins = [ins.get("title", "") for ins in insights_list[:3]]
            
            entry_id = save_history_entry(
                dataset_name=st.session_state.get("data_source", "Unknown"),
                record_count=len(current_df) if current_df is not None else 0,
                column_count=len(current_df.columns) if current_df is not None else 0,
                audit_log=audit_log,
                insights_summary=top_ins
            )
            if entry_id:
                st.success(f"Snapshot saved successfully! Entry ID: `{entry_id}`")
                st.rerun()
                
    with hist_col2:
        if st.button("🗑️ Clear All Saved History"):
            deleted = clear_all_history()
            st.success(f"Purged {deleted} history records.")
            st.rerun()
            
    st.markdown("---")
    
    history_entries = load_history_entries()
    if not history_entries:
        st.info("No saved analysis history records found. Click 'Save Current Analysis Snapshot' above to create one.")
    else:
        st.markdown(f"**Saved Snapshots ({len(history_entries)}):**")
        
        for entry in history_entries:
            with st.expander(f"🕒 {entry.get('timestamp')} — {entry.get('dataset_name')} ({entry.get('record_count')} records)"):
                e_col1, e_col2, e_col3, e_col4 = st.columns(4)
                e_col1.write(f"**Records:** `{entry.get('record_count')}`")
                e_col2.write(f"**Columns:** `{entry.get('column_count')}`")
                e_col3.write(f"**Duplicates Purged:** `{entry.get('duplicates_removed')}`")
                e_col4.write(f"**Values Imputed:** `{entry.get('values_imputed')}`")
                
                if entry.get("top_insights"):
                    st.write("**Key Insights Recorded:**")
                    for ins_t in entry["top_insights"]:
                        st.write(f"• {ins_t}")
                        
                if st.button(f"🗑️ Delete Entry {entry.get('entry_id')}", key=f"del_{entry.get('entry_id')}"):
                    delete_history_entry(entry.get("entry_id"))
                    st.rerun()
                    
        st.markdown("---")
        hist_json = export_history_json()
        st.download_button(
            label="📥 Export All History Snapshots (.json)",
            data=hist_json.encode('utf-8'),
            file_name="analysis_history_export.json",
            mime="application/json"
        )
