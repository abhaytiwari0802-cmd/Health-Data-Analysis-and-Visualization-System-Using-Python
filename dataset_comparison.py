"""
Module: dataset_comparison.py
Description: Compares two health datasets or two stages of a dataset (e.g. Raw vs Cleaned).
             Evaluates dimensional changes, schema overlap, missingness deltas,
             and statistical drift for shared numerical and categorical variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple


def compare_datasets(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Dataset A (Raw)",
    label_b: str = "Dataset B (Cleaned)"
) -> Dict[str, Any]:
    """
    Computes a comprehensive statistical and structural comparison between two datasets.
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return {
            "error": "One or both datasets are empty or None.",
            "label_a": label_a,
            "label_b": label_b
        }
        
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    
    shared_cols = sorted(list(cols_a.intersection(cols_b)))
    only_in_a = sorted(list(cols_a - cols_b))
    only_in_b = sorted(list(cols_b - cols_a))
    
    # Missing cells comparison
    missing_a = int(df_a.isna().sum().sum())
    missing_b = int(df_b.isna().sum().sum())
    total_cells_a = len(df_a) * len(df_a.columns)
    total_cells_b = len(df_b) * len(df_b.columns)
    
    missing_pct_a = round((missing_a / total_cells_a) * 100, 2) if total_cells_a > 0 else 0.0
    missing_pct_b = round((missing_b / total_cells_b) * 100, 2) if total_cells_b > 0 else 0.0
    
    # Duplicate rows
    dups_a = int(df_a.duplicated().sum())
    dups_b = int(df_b.duplicated().sum())
    
    # Numeric column statistics comparison for shared columns
    stats_comparison = []
    for col in shared_cols:
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            s_a = df_a[col].dropna()
            s_b = df_b[col].dropna()
            
            mean_a = round(float(s_a.mean()), 2) if not s_a.empty else 0.0
            mean_b = round(float(s_b.mean()), 2) if not s_b.empty else 0.0
            
            median_a = round(float(s_a.median()), 2) if not s_a.empty else 0.0
            median_b = round(float(s_b.median()), 2) if not s_b.empty else 0.0
            
            std_a = round(float(s_a.std()), 2) if len(s_a) > 1 else 0.0
            std_b = round(float(s_b.std()), 2) if len(s_b) > 1 else 0.0
            
            stats_comparison.append({
                "Parameter": col,
                f"{label_a} Mean": mean_a,
                f"{label_b} Mean": mean_b,
                "Delta Mean": round(mean_b - mean_a, 2),
                f"{label_a} Median": median_a,
                f"{label_b} Median": median_b,
                f"{label_a} Std": std_a,
                f"{label_b} Std": std_b
            })
            
    stats_df = pd.DataFrame(stats_comparison)
    
    # Missing counts per shared column
    missing_col_comparison = []
    for col in shared_cols:
        null_a = int(df_a[col].isna().sum())
        null_b = int(df_b[col].isna().sum())
        missing_col_comparison.append({
            "Column": col,
            f"{label_a} Missing": null_a,
            f"{label_b} Missing": null_b,
            "Resolved": null_a - null_b
        })
    missing_cols_df = pd.DataFrame(missing_col_comparison)
    
    return {
        "label_a": label_a,
        "label_b": label_b,
        "rows_a": len(df_a),
        "rows_b": len(df_b),
        "delta_rows": len(df_b) - len(df_a),
        "cols_a": len(df_a.columns),
        "cols_b": len(df_b.columns),
        "shared_columns_count": len(shared_cols),
        "shared_columns": shared_cols,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "missing_cells_a": missing_a,
        "missing_cells_b": missing_b,
        "missing_pct_a": missing_pct_a,
        "missing_pct_b": missing_pct_b,
        "duplicates_a": dups_a,
        "duplicates_b": dups_b,
        "stats_comparison": stats_df,
        "missing_col_comparison": missing_cols_df
    }


def plot_dataset_comparison(comp_data: Dict[str, Any]) -> plt.Figure:
    """
    Renders visual comparison charts for rows, duplicates, and missing values.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    label_a = comp_data.get("label_a", "A")
    label_b = comp_data.get("label_b", "B")
    
    # 1. Row Counts
    rows = [comp_data.get("rows_a", 0), comp_data.get("rows_b", 0)]
    bars1 = axes[0].bar([label_a, label_b], rows, color=["#3B82F6", "#10B981"], width=0.5)
    axes[0].set_title("Total Records", fontweight="bold", fontsize=11)
    for b in bars1:
        axes[0].annotate(f"{b.get_height()}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
                         
    # 2. Missing Value Percentages
    miss_pcts = [comp_data.get("missing_pct_a", 0.0), comp_data.get("missing_pct_b", 0.0)]
    bars2 = axes[1].bar([label_a, label_b], miss_pcts, color=["#F59E0B", "#10B981"], width=0.5)
    axes[1].set_title("Missing Cells (%)", fontweight="bold", fontsize=11)
    for b in bars2:
        axes[1].annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
                         
    # 3. Duplicate Records
    dups = [comp_data.get("duplicates_a", 0), comp_data.get("duplicates_b", 0)]
    bars3 = axes[2].bar([label_a, label_b], dups, color=["#EF4444", "#10B981"], width=0.5)
    axes[2].set_title("Duplicate Rows", fontweight="bold", fontsize=11)
    for b in bars3:
        axes[2].annotate(f"{b.get_height()}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
                         
    plt.tight_layout()
    return fig
