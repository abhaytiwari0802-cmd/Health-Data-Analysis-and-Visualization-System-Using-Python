"""
Module: visualization.py
Description: Generates publication-ready Matplotlib and Seaborn visualizations
             for patient demographics, vital signs, correlations, and clinical risks.
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend for web & multi-threaded safety
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from modules.statistical_analysis import (
    categorize_age_groups,
    categorize_bmi,
    calculate_pearson_correlation_numpy,
    calculate_spearman_correlation_numpy,
    stratify_clinical_risks
)

PRIMARY_COLOR = "#1E3A8A"      # Deep Navy
SECONDARY_COLOR = "#0D9488"    # Medical Teal
ACCENT_COLOR = "#E11D48"       # Crimson Alert

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "grid.color": "#F1F5F9",
    "grid.linestyle": "--"
})


def plot_age_distribution(df: pd.DataFrame, bins: int = 15) -> plt.Figure:
    """Plots an Age distribution histogram with KDE and mean/median indicators."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "Age" not in df.columns or df["Age"].dropna().empty:
        ax.text(0.5, 0.5, "Age data unavailable", ha="center", va="center")
        return fig
        
    ages = df["Age"].dropna().values
    mean_age = float(np.mean(ages))
    median_age = float(np.median(ages))
    
    sns.histplot(ages, bins=bins, kde=True, color=PRIMARY_COLOR, ax=ax, edgecolor="white", alpha=0.75)
    ax.axvline(mean_age, color="#DC2626", linestyle="--", linewidth=2, label=f"Mean: {mean_age:.1f} yrs")
    ax.axvline(median_age, color="#059669", linestyle=":", linewidth=2, label=f"Median: {median_age:.1f} yrs")
    
    ax.set_title("Patient Age Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Age (years)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    return fig


def plot_gender_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots a Gender distribution bar chart with value counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if "Gender" not in df.columns or df["Gender"].dropna().empty:
        ax.text(0.5, 0.5, "Gender data unavailable", ha="center", va="center")
        return fig
        
    counts = df["Gender"].value_counts()
    total = len(df["Gender"].dropna())
    
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, "#F59E0B", "#8B5CF6"][:len(counts)]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.55)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100 if total > 0 else 0
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
                    
    ax.set_title("Patient Gender Breakdown", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.25 if not counts.empty else 10)
    plt.tight_layout()
    return fig


def plot_age_group_distribution(
    df: pd.DataFrame,
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None
) -> plt.Figure:
    """Plots the distribution of patients across demographic age groups."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "Age" not in df.columns or df["Age"].dropna().empty:
        ax.text(0.5, 0.5, "Age data unavailable", ha="center", va="center")
        return fig
        
    groups = categorize_age_groups(df, bins=bins, labels=labels)
    counts = groups.value_counts(sort=False)
    
    palette = sns.color_palette("Blues_r", n_colors=len(counts))
    bars = ax.bar(counts.index.astype(str), counts.values, color=palette, width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100 if len(df) > 0 else 0
        ax.annotate(f"{height} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    ax.set_title("Demographic Age-Group Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylim(0, max(counts.values) * 1.25 if not counts.empty and max(counts.values) > 0 else 10)
    plt.tight_layout()
    return fig


def plot_bmi_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots a BMI histogram with color-coded WHO clinical classification zones."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if "BMI" not in df.columns or df["BMI"].dropna().empty:
        ax.text(0.5, 0.5, "BMI data unavailable", ha="center", va="center")
        return fig
        
    bmis = df["BMI"].dropna().values
    
    ax.axvspan(0, 18.5, color="#93C5FD", alpha=0.25, label="Underweight (<18.5)")
    ax.axvspan(18.5, 24.9, color="#86EFAC", alpha=0.25, label="Normal (18.5-24.9)")
    ax.axvspan(25.0, 29.9, color="#FDE047", alpha=0.25, label="Overweight (25-29.9)")
    ax.axvspan(30.0, 60.0, color="#FCA5A5", alpha=0.25, label="Obese (≥30.0)")
    
    sns.histplot(bmis, bins=18, color="#1E293B", ax=ax, edgecolor="white", alpha=0.85)
    mean_bmi = float(np.mean(bmis))
    ax.axvline(mean_bmi, color="#B91C1C", linestyle="--", linewidth=2, label=f"Mean BMI: {mean_bmi:.1f}")
    
    ax.set_title("BMI Distribution & WHO Categories", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Body Mass Index (kg/m²)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_xlim(min(14.0, float(np.min(bmis)) - 1), max(45.0, float(np.max(bmis)) + 2))
    ax.legend(frameon=True, facecolor="white", loc="upper right", fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_demographics_overview(df: pd.DataFrame) -> plt.Figure:
    """Combines Age, Gender, Age Group, and BMI into a 2x2 presentation dashboard grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Age
    if "Age" in df.columns and not df["Age"].dropna().empty:
        ages = df["Age"].dropna().values
        sns.histplot(ages, bins=14, kde=True, color=PRIMARY_COLOR, ax=axes[0, 0], edgecolor="white", alpha=0.7)
        axes[0, 0].axvline(float(np.mean(ages)), color="red", linestyle="--", label=f"Mean: {float(np.mean(ages)):.1f}")
        axes[0, 0].set_title("A. Age Distribution", fontweight="bold", fontsize=12)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "Age Data Unavailable", ha="center")
        
    # 2. Gender
    if "Gender" in df.columns and not df["Gender"].dropna().empty:
        g_counts = df["Gender"].value_counts()
        axes[0, 1].pie(g_counts.values, labels=g_counts.index, autopct="%1.1f%%",
                       colors=[PRIMARY_COLOR, SECONDARY_COLOR, "#F59E0B"][:len(g_counts)],
                       startangle=140, explode=[0.03]*len(g_counts))
        axes[0, 1].set_title("B. Gender Proportion", fontweight="bold", fontsize=12)
    else:
        axes[0, 1].text(0.5, 0.5, "Gender Data Unavailable", ha="center")
        
    # 3. Age Group
    if "Age" in df.columns and not df["Age"].dropna().empty:
        grp = categorize_age_groups(df).value_counts(sort=False)
        sns.barplot(x=grp.index.astype(str), y=grp.values, ax=axes[1, 0], hue=grp.index.astype(str), palette="Blues_r", legend=False)
        axes[1, 0].set_title("C. Age Group Cohorts", fontweight="bold", fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=25)
    else:
        axes[1, 0].text(0.5, 0.5, "Age Groups Unavailable", ha="center")
        
    # 4. BMI Categories
    if "BMI" in df.columns and not df["BMI"].dropna().empty:
        bmi_grp = categorize_bmi(df).value_counts(sort=False)
        colors = ["#60A5FA", "#34D399", "#FBBF24", "#F87171"][:len(bmi_grp)]
        sns.barplot(x=bmi_grp.index.astype(str), y=bmi_grp.values, ax=axes[1, 1], hue=bmi_grp.index.astype(str), palette=colors, legend=False)
        axes[1, 1].set_title("D. WHO BMI Categories", fontweight="bold", fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=15)
    else:
        axes[1, 1].text(0.5, 0.5, "BMI Data Unavailable", ha="center")
        
    plt.tight_layout()
    return fig


def plot_vital_signs_boxplots(df: pd.DataFrame, columns: Optional[List[str]] = None) -> plt.Figure:
    """Renders standardized box plots to visualize distribution and outliers across vitals."""
    default_vitals = ["Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Glucose", "Cholesterol", "Hemoglobin"]
    target_cols = [c for c in (columns or default_vitals) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if not target_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No numeric vital sign columns available", ha="center")
        return fig
        
    n_cols = len(target_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(max(12, n_cols * 2.8), 5))
    if n_cols == 1:
        axes = [axes]
        
    for i, col in enumerate(target_cols):
        clean_vals = df[col].dropna()
        sns.boxplot(y=clean_vals, ax=axes[i], color=SECONDARY_COLOR, width=0.45, fliersize=4)
        axes[i].set_title(col.replace("_", " "), fontsize=11, fontweight="bold")
        axes[i].set_ylabel("")
        
    plt.suptitle("Vital Signs & Clinical Biomarker Spread (Box Plots)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_vital_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None) -> plt.Figure:
    """Creates distribution histograms with KDE overlays for key health indicators."""
    default_vitals = ["Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Glucose", "Cholesterol", "Hemoglobin"]
    target_cols = [c for c in (columns or default_vitals) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if not target_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No numeric vital sign columns available", ha="center")
        return fig
        
    n_plots = len(target_cols)
    n_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 3.5))
    axes = np.array(axes).reshape(-1)
    
    for i, col in enumerate(target_cols):
        clean_vals = df[col].dropna()
        sns.histplot(clean_vals, kde=True, ax=axes[i], color=PRIMARY_COLOR, edgecolor="white", alpha=0.7)
        mean_v = float(np.mean(clean_vals))
        axes[i].axvline(mean_v, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.1f}")
        axes[i].set_title(f"{col.replace('_', ' ')} Distribution", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col.replace("_", " "))
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=8)
        
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame, method_name: str = "Pearson r") -> plt.Figure:
    """Renders an annotated Seaborn heatmap for correlation matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if corr_df.empty:
        ax.text(0.5, 0.5, "Correlation matrix is empty", ha="center")
        return fig
        
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.75,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": f"{method_name} Correlation"},
        ax=ax
    )
    
    ax.set_title(f"Biomedical Correlation Heatmap ({method_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=35, ha="right", fontsize=9.5)
    plt.yticks(fontsize=9.5)
    plt.tight_layout()
    return fig


def plot_scatter_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = "Gender",
    method: str = "pearson"
) -> Tuple[plt.Figure, float, int]:
    """
    Scatter plot between two health indicators with linear trendline and correlation value.
    Returns (Figure, correlation_val, valid_N).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if x_col not in df.columns or y_col not in df.columns:
        ax.text(0.5, 0.5, f"Columns {x_col} or {y_col} not found", ha="center")
        return fig, 0.0, 0
        
    valid_data = df[[x_col, y_col] + ([hue_col] if hue_col and hue_col in df.columns else [])].dropna()
    x = valid_data[x_col].values
    y = valid_data[y_col].values
    
    if method == "spearman":
        r_val, n_obs, _ = calculate_spearman_correlation_numpy(x, y)
        metric_label = f"Spearman rho = {r_val:+.3f}"
    else:
        r_val, n_obs, _ = calculate_pearson_correlation_numpy(x, y)
        metric_label = f"Pearson r = {r_val:+.3f}"
        
    if hue_col and hue_col in valid_data.columns:
        sns.scatterplot(data=valid_data, x=x_col, y=y_col, hue=hue_col, palette="Set1", alpha=0.85, s=65, ax=ax)
    else:
        sns.scatterplot(data=valid_data, x=x_col, y=y_col, color=PRIMARY_COLOR, alpha=0.85, s=65, ax=ax)
        
    if len(x) > 2 and np.std(x) > 0:
        poly_coeff = np.polyfit(x, y, 1)
        x_trend = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_trend = np.polyval(poly_coeff, x_trend)
        ax.plot(x_trend, y_trend, color="#DC2626", linestyle="--", linewidth=2,
                label=f"Trendline ({metric_label}, N={n_obs})")
                
    ax.set_title(f"Relationship: {x_col.replace('_', ' ')} vs {y_col.replace('_', ' ')} (N={n_obs})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(x_col.replace("_", " "), fontsize=11)
    ax.set_ylabel(y_col.replace("_", " "), fontsize=11)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    return fig, r_val, n_obs


def plot_clinical_risk_charts(df: pd.DataFrame) -> plt.Figure:
    """Plots stacked/categorical clinical risk categories for Blood Pressure, Glucose, and Cholesterol."""
    risks = stratify_clinical_risks(df)
    n_risks = len(risks)
    
    if n_risks == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Insufficient clinical columns to evaluate clinical risk categories", ha="center")
        return fig
        
    fig, axes = plt.subplots(1, n_risks, figsize=(max(12, n_risks * 4.5), 4.5))
    if n_risks == 1:
        axes = [axes]
        
    palette_map = {
        "BP_Category": ["#10B981", "#F59E0B", "#F97316", "#EF4444"],
        "Glucose_Category": ["#10B981", "#F59E0B", "#EF4444"],
        "Cholesterol_Category": ["#10B981", "#F59E0B", "#EF4444"]
    }
    
    for i, (k, series) in enumerate(risks.items()):
        counts = series.value_counts()
        colors = palette_map.get(k, sns.color_palette("Set2"))[:len(counts)]
        
        bars = axes[i].bar(counts.index.astype(str), counts.values, color=colors, width=0.55)
        for b in bars:
            h = b.get_height()
            pct = (h / len(df)) * 100 if len(df) > 0 else 0
            axes[i].annotate(f"{h}\n({pct:.1f}%)",
                             xy=(b.get_x() + b.get_width() / 2, h),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                             
        title = k.replace("_", " ")
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].set_ylabel("Patient Count")
        axes[i].set_ylim(0, max(counts.values) * 1.25 if not counts.empty else 10)
        
    plt.tight_layout()
    return fig
