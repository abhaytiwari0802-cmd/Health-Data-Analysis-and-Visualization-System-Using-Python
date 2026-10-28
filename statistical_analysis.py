"""
Module: statistical_analysis.py
Description: Computes descriptive statistics, demographic groupings,
             clinical risk stratifications, Pearson & Spearman correlation matrices using NumPy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_AGE_BINS = [0, 17, 35, 50, 65, 120]
DEFAULT_AGE_LABELS = [
    "Pediatric (0-17)",
    "Young Adult (18-35)",
    "Adult (36-50)",
    "Middle-Aged (51-65)",
    "Senior (>65)"
]

BMI_BINS = [0.0, 18.49, 24.99, 29.99, 100.0]
BMI_LABELS = ["Underweight", "Normal weight", "Overweight", "Obese"]


def calculate_descriptive_stats_numpy(arr: np.ndarray) -> Dict[str, Any]:
    """
    Computes full descriptive statistics on a 1D NumPy array.
    Includes Mean, Median, Mode, Std Dev, Variance, Min, Max, Range, Q1, Q3, and IQR.
    """
    clean_arr = arr[~np.isnan(arr)]
    n = len(clean_arr)
    
    if n == 0:
        return {
            "Count": 0, "Mean": 0.0, "Median": 0.0, "Mode": 0.0, "Std_Dev": 0.0,
            "Variance": 0.0, "Min": 0.0, "Max": 0.0, "Range": 0.0,
            "Q1_25%": 0.0, "Q3_75%": 0.0, "IQR": 0.0
        }
        
    mean_val = float(np.mean(clean_arr))
    median_val = float(np.median(clean_arr))
    
    # Calculate mode using pandas Series on clean array
    s = pd.Series(clean_arr)
    mode_series = s.mode()
    mode_val = round(float(mode_series[0]), 2) if not mode_series.empty else median_val
    
    std_val = float(np.std(clean_arr, ddof=1)) if n > 1 else 0.0
    var_val = float(np.var(clean_arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(clean_arr))
    max_val = float(np.max(clean_arr))
    range_val = max_val - min_val
    q1_val = float(np.percentile(clean_arr, 25))
    q3_val = float(np.percentile(clean_arr, 75))
    iqr_val = q3_val - q1_val
    
    return {
        "Count": int(n),
        "Mean": round(mean_val, 2),
        "Median": round(median_val, 2),
        "Mode": mode_val,
        "Std_Dev": round(std_val, 2),
        "Variance": round(var_val, 2),
        "Min": round(min_val, 2),
        "Max": round(max_val, 2),
        "Range": round(range_val, 2),
        "Q1_25%": round(q1_val, 2),
        "Q3_75%": round(q3_val, 2),
        "IQR": round(iqr_val, 2)
    }


def generate_dataset_summary_statistics(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generates a descriptive statistical summary table for numeric columns.
    """
    if target_columns is None:
        target_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
    summary_data = []
    for col in target_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            arr = df[col].to_numpy(dtype=np.float64, copy=True)
            stats = calculate_descriptive_stats_numpy(arr)
            stats["Variable"] = col
            summary_data.append(stats)
            
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        cols = ["Variable"] + [c for c in summary_df.columns if c != "Variable"]
        summary_df = summary_df[cols]
    return summary_df


def categorize_age_groups(
    df: pd.DataFrame,
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None
) -> pd.Series:
    """
    Categorizes patient ages into discrete demographic groups.
    """
    if "Age" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)
        
    age_bins = bins if bins is not None else DEFAULT_AGE_BINS
    age_labels = labels if labels is not None else DEFAULT_AGE_LABELS
    return pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True, include_lowest=True)


def categorize_bmi(df: pd.DataFrame) -> pd.Series:
    """
    Categorizes patients into WHO BMI categories.
    """
    if "BMI" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)
        
    return pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=True, include_lowest=True)


def calculate_pearson_correlation_numpy(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, str]:
    """
    Calculates Pearson Correlation Coefficient (r) and returns (r_value, valid_n, message).
    """
    valid_mask = (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    n = len(x_clean)
    
    if n < 2:
        return 0.0, n, "Insufficient data points (< 2) for correlation calculation."
        
    x_diff = x_clean - np.mean(x_clean)
    y_diff = y_clean - np.mean(y_clean)
    
    sum_sq_x = np.sum(x_diff ** 2)
    sum_sq_y = np.sum(y_diff ** 2)
    
    if sum_sq_x == 0.0 or sum_sq_y == 0.0:
        return 0.0, n, "Zero variance in one or both variables; correlation is undefined."
        
    numerator = np.sum(x_diff * y_diff)
    denominator = np.sqrt(sum_sq_x * sum_sq_y)
    r = float(numerator / denominator)
    r = max(-1.0, min(1.0, r))
    
    return round(r, 4), n, "Success"


def calculate_spearman_correlation_numpy(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, str]:
    """
    Calculates Spearman Rank Correlation Coefficient (rho).
    Computes Pearson correlation on the fractional ranks of the variables.
    """
    valid_mask = (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    n = len(x_clean)
    
    if n < 2:
        return 0.0, n, "Insufficient observations (< 2) for Spearman correlation."
        
    # Compute ranks (with average rank for ties)
    x_rank = pd.Series(x_clean).rank().to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_clean).rank().to_numpy(dtype=np.float64)
    
    r, _, msg = calculate_pearson_correlation_numpy(x_rank, y_rank)
    return r, n, msg


def compute_correlation_matrix_numpy(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes pairwise correlation matrix using either 'pearson' or 'spearman' method.
    Handles constant or invalid columns gracefully.
    """
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
    valid_cols = []
    warnings = []
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        arr = df[col].to_numpy(dtype=np.float64, copy=True)
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) > 1 and np.std(arr_clean) > 0:
            valid_cols.append(col)
        else:
            warnings.append(f"Column '{col}' omitted from correlation matrix (constant or empty values).")
            
    n = len(valid_cols)
    matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                x_vals = df[valid_cols[i]].values
                y_vals = df[valid_cols[j]].values
                if method == "spearman":
                    r, _, _ = calculate_spearman_correlation_numpy(x_vals, y_vals)
                else:
                    r, _, _ = calculate_pearson_correlation_numpy(x_vals, y_vals)
                    
                matrix[i, j] = r
                matrix[j, i] = r
                
    corr_df = pd.DataFrame(matrix, index=valid_cols, columns=valid_cols)
    return corr_df, warnings


def interpret_correlation_strength(r: float, method_name: str = "Pearson r") -> Dict[str, str]:
    """
    Provides standard academic interpretation of correlation strength and direction.
    """
    abs_r = abs(r)
    
    if abs_r >= 0.70:
        strength = "Strong"
    elif abs_r >= 0.40:
        strength = "Moderate"
    elif abs_r >= 0.20:
        strength = "Weak"
    else:
        strength = "Negligible / Very Weak"
        
    if r > 0.05:
        direction = "Positive (both variables increase together)"
    elif r < -0.05:
        direction = "Negative (one variable increases as the other decreases)"
    else:
        direction = "Neutral / No linear relationship"
        
    explanation = (
        f"{method_name} = {r:.3f} indicates a {strength.lower()} {direction.lower()}. "
        "IMPORTANT: Correlation indicates statistical association, NOT clinical causation."
    )
    
    return {
        "r_value": f"{r:.3f}",
        "strength": strength,
        "direction": direction,
        "explanation": explanation
    }


def stratify_clinical_risks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Stratifies patients into standard medical risk categories:
    - Blood Pressure: AHA 2017 Guidelines
    - Blood Glucose: ADA Fasting Guidelines
    - Cholesterol: NCEP ATP III Guidelines
    """
    results = {}
    
    if "Systolic_BP" in df.columns and "Diastolic_BP" in df.columns:
        sys = df["Systolic_BP"].values
        dia = df["Diastolic_BP"].values
        
        conditions = [
            (sys >= 140) | (dia >= 90),
            (sys >= 130) | (dia >= 80),
            (sys >= 120) & (dia < 80),
            (sys < 120) & (dia < 80)
        ]
        choices = [
            "Hypertension Stage 2",
            "Hypertension Stage 1",
            "Elevated BP",
            "Normal BP"
        ]
        results["BP_Category"] = pd.Series(np.select(conditions, choices, default="Unclassified"), index=df.index)
        
    if "Blood_Glucose" in df.columns:
        gluc = df["Blood_Glucose"].values
        conditions = [
            gluc >= 126.0,
            (gluc >= 100.0) & (gluc < 126.0),
            gluc < 100.0
        ]
        choices = [
            "Diabetic Range (≥126 mg/dL)",
            "Prediabetic Range (100-125 mg/dL)",
            "Normal Range (<100 mg/dL)"
        ]
        results["Glucose_Category"] = pd.Series(np.select(conditions, choices, default="Unclassified"), index=df.index)
        
    if "Cholesterol" in df.columns:
        chol = df["Cholesterol"].values
        conditions = [
            chol >= 240.0,
            (chol >= 200.0) & (chol < 240.0),
            chol < 200.0
        ]
        choices = [
            "High (≥240 mg/dL)",
            "Borderline High (200-239 mg/dL)",
            "Desirable (<200 mg/dL)"
        ]
        results["Cholesterol_Category"] = pd.Series(np.select(conditions, choices, default="Unclassified"), index=df.index)
        
    return results
