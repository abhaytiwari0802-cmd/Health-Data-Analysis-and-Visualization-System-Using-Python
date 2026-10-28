"""
Module: insights.py
Description: Rule-based automated analytical insight generator for health datasets.
             Uses real statistical calculations and clinical benchmarks without external AI APIs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from modules.statistical_analysis import (
    compute_correlation_matrix_numpy,
    stratify_clinical_risks,
    categorize_age_groups,
    categorize_bmi
)


def generate_automated_insights(
    df: pd.DataFrame,
    audit_log: Optional[Dict[str, Any]] = None,
    raw_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Generates rule-based analytical insights from actual dataset statistics.
    Every observation is backed by real mathematical computation.
    """
    insights = []
    
    if df is None or df.empty:
        return [{
            "title": "Dataset Empty",
            "category": "Data Quality",
            "badge": "warning",
            "observation": "No patient records are currently available for statistical insight generation.",
            "methodology": "Checked len(df) == 0."
        }]
        
    n_rows = len(df)
    n_cols = len(df.columns)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # -----------------------------------------------------------------------
    # 1. Dataset Dimensions & Completeness
    # -----------------------------------------------------------------------
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    completeness_pct = round(((total_cells - missing_cells) / total_cells) * 100, 1) if total_cells > 0 else 0
    
    insights.append({
        "title": "Dataset Completeness & Scale",
        "category": "Data Quality",
        "badge": "success" if completeness_pct >= 95 else "warning",
        "observation": f"The active cohort contains {n_rows:,} patient records across {n_cols} attributes ({len(numeric_cols)} numerical, {n_cols - len(numeric_cols)} categorical). Overall data completeness is {completeness_pct}%.",
        "methodology": "Calculated total non-null cells divided by total grid cells (Rows x Columns)."
    })
    
    # -----------------------------------------------------------------------
    # 2. Cleaning & Preprocessing Impact
    # -----------------------------------------------------------------------
    if audit_log:
        dups_removed = audit_log.get("duplicates_removed", 0)
        imputed_dict = audit_log.get("imputed_details", {})
        total_imputed = sum(d["imputed_count"] for d in imputed_dict.values() if isinstance(d, dict) and "imputed_count" in d)
        
        if dups_removed > 0 or total_imputed > 0:
            insights.append({
                "title": "NumPy Preprocessing Impact",
                "category": "Data Quality",
                "badge": "info",
                "observation": f"Data cleaning resolved {dups_removed} duplicate patient records and imputed {total_imputed} missing entries across {len(imputed_dict)} attributes using {audit_log.get('imputation_strategy', 'median')} imputation.",
                "methodology": "Audited changes recorded during perform_full_cleaning_pipeline execution."
            })
            
    # -----------------------------------------------------------------------
    # 3. Missing Value Hotspots
    # -----------------------------------------------------------------------
    col_nulls = df.isna().sum()
    hotspots = col_nulls[col_nulls > 0].sort_values(ascending=False)
    if not hotspots.empty:
        worst_col = hotspots.index[0]
        worst_pct = round((hotspots.iloc[0] / n_rows) * 100, 1)
        insights.append({
            "title": "Missing Data Concentrations",
            "category": "Data Quality",
            "badge": "warning",
            "observation": f"Attribute '{worst_col}' has the highest missing rate ({hotspots.iloc[0]} missing cells, {worst_pct}%). Imputation or targeted clinical follow-up is recommended.",
            "methodology": "Computed column-wise null percentages using df.isna().sum() / len(df)."
        })
        
    # -----------------------------------------------------------------------
    # 4. Distribution Skewness & Central Tendency
    # -----------------------------------------------------------------------
    skewed_observations = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) >= 10:
            mean_v = vals.mean()
            median_v = vals.median()
            std_v = vals.std()
            if std_v > 0:
                # Non-parametric skew measure: (Mean - Median) / Std
                skew_ratio = (mean_v - median_v) / std_v
                if skew_ratio > 0.25:
                    skewed_observations.append(f"'{col}' shows right skew (Mean={mean_v:.1f} > Median={median_v:.1f}), indicating a subset of patients with elevated values.")
                elif skew_ratio < -0.25:
                    skewed_observations.append(f"'{col}' shows left skew (Mean={mean_v:.1f} < Median={median_v:.1f}).")
                    
    if skewed_observations:
        insights.append({
            "title": "Biomarker Distribution Skewness",
            "category": "Distribution",
            "badge": "info",
            "observation": " ".join(skewed_observations[:2]),
            "methodology": "Evaluated Pearson's non-parametric skewness coefficient: (Mean - Median) / Standard Deviation."
        })
        
    # -----------------------------------------------------------------------
    # 5. Key Correlation Patterns
    # -----------------------------------------------------------------------
    if len(numeric_cols) >= 2:
        corr_m, _ = compute_correlation_matrix_numpy(df, columns=numeric_cols, method="pearson")
        strong_pairs = []
        cols = corr_m.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val = corr_m.iloc[i, j]
                if abs(r_val) >= 0.35:
                    strong_pairs.append((cols[i], cols[j], r_val))
                    
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if strong_pairs:
            top = strong_pairs[0]
            direction = "positive" if top[2] > 0 else "negative"
            strength = "strong" if abs(top[2]) >= 0.6 else "moderate"
            insights.append({
                "title": "Primary Biomedical Correlation",
                "category": "Correlation",
                "badge": "primary",
                "observation": f"The strongest observed linear relationship is between '{top[0]}' and '{top[1]}' (Pearson r = {top[2]:+.3f}, {strength} {direction}). As {top[0]} increases, {top[1]} tends to {'increase' if top[2] > 0 else 'decrease'}.",
                "methodology": "Pairwise Pearson correlation computed across all available continuous features using NumPy array products."
            })
            
    # -----------------------------------------------------------------------
    # 6. Clinical Risk Stratifications (AHA / ADA Guidelines)
    # -----------------------------------------------------------------------
    risks = stratify_clinical_risks(df)
    if "BP_Category" in risks:
        bp_series = risks["BP_Category"]
        htn_count = int(bp_series.isin(["Hypertension Stage 1", "Hypertension Stage 2"]).sum())
        htn_pct = round((htn_count / n_rows) * 100, 1)
        insights.append({
            "title": "Hypertension Prevalence (AHA 2017)",
            "category": "Clinical Risk",
            "badge": "warning" if htn_pct >= 30 else "info",
            "observation": f"{htn_count} patients ({htn_pct}%) meet criteria for Stage 1 or Stage 2 Hypertension based on systolic (≥130 mmHg) or diastolic (≥80 mmHg) readings.",
            "methodology": "Categorized systolic and diastolic blood pressure pairs against AHA/ACC 2017 diagnostic stages."
        })
        
    if "Glucose_Category" in risks:
        glu_series = risks["Glucose_Category"]
        diab_count = int((glu_series == "Diabetic Range (≥126 mg/dL)").sum())
        diab_pct = round((diab_count / n_rows) * 100, 1)
        if diab_count > 0:
            insights.append({
                "title": "Elevated Fasting Glucose (ADA Criteria)",
                "category": "Clinical Risk",
                "badge": "warning",
                "observation": f"{diab_count} patients ({diab_pct}%) present fasting blood glucose levels in the diabetic range (≥126 mg/dL).",
                "methodology": "Applied American Diabetes Association fasting plasma glucose thresholds."
            })
            
    # -----------------------------------------------------------------------
    # 7. Subgroup Demographic Comparison
    # -----------------------------------------------------------------------
    if "Gender" in df.columns and "Systolic_BP" in df.columns:
        gender_bp = df.groupby("Gender")["Systolic_BP"].mean().dropna()
        if len(gender_bp) >= 2:
            g_str = ", ".join([f"{g}: {m:.1f} mmHg" for g, m in gender_bp.items()])
            insights.append({
                "title": "Demographic Subgroup Contrast",
                "category": "Demographics",
                "badge": "info",
                "observation": f"Mean Systolic Blood Pressure by gender group: {g_str}.",
                "methodology": "Calculated group-specific arithmetic means via df.groupby('Gender')['Systolic_BP'].mean()."
            })
            
    return insights
