"""
Module: data_cleaning.py
Description: Performs comprehensive data quality analysis and data cleaning operations.
             Includes missing value imputation via NumPy, duplicate elimination,
             IQR statistical outlier detection & treatment, biomedical range validation,
             vectorized BMI calculation, and detailed audit logging.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional

# Medically grounded reference bounds for validation
BIOMEDICAL_BOUNDS = {
    "Age": (1.0, 115.0),
    "Height_cm": (50.0, 240.0),
    "Weight_kg": (20.0, 250.0),
    "BMI": (10.0, 70.0),
    "Systolic_BP": (60.0, 240.0),
    "Diastolic_BP": (40.0, 150.0),
    "Heart_Rate": (35.0, 200.0),
    "Blood_Glucose": (40.0, 450.0),
    "Cholesterol": (80.0, 500.0),
    "Hemoglobin": (5.0, 22.0)
}


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects missing values across all columns safely.
    Uses pd.api.types.is_numeric_dtype to avoid TypeError with pandas ExtensionDtypes.
    Returns a DataFrame summary with missing counts and percentages.
    """
    summary = []
    total_rows = len(df)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            arr = df[col].to_numpy(dtype=np.float64, copy=True)
            nan_count = int(np.sum(np.isnan(arr)))
        else:
            nan_count = int(df[col].isna().sum())
            
        pct = round((nan_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        summary.append({
            "Column": col,
            "Missing Count": nan_count,
            "Missing (%)": pct,
            "Has Missing": nan_count > 0
        })
        
    return pd.DataFrame(summary)


def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies duplicate records in the dataset.
    Checks both full-row duplicates and duplicate Patient_IDs if present.
    """
    full_duplicates_count = int(df.duplicated().sum())
    full_duplicates_mask = df.duplicated(keep=False)
    
    pid_duplicates_count = 0
    if "Patient_ID" in df.columns:
        pid_duplicates_count = int(df["Patient_ID"].duplicated().sum())
        
    return {
        "full_duplicates_count": full_duplicates_count,
        "pid_duplicates_count": pid_duplicates_count,
        "has_duplicates": (full_duplicates_count > 0 or pid_duplicates_count > 0),
        "duplicate_rows": df[full_duplicates_mask].copy() if full_duplicates_count > 0 else pd.DataFrame()
    }


def remove_duplicates(df: pd.DataFrame, drop_by_patient_id: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate records from the dataset.
    Preserves the first occurrence and logs removed count.
    """
    initial_count = len(df)
    cleaned_df = df.drop_duplicates(keep="first").copy()
    
    if drop_by_patient_id and "Patient_ID" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=["Patient_ID"], keep="first").copy()
        
    removed_count = initial_count - len(cleaned_df)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df, removed_count


def handle_missing_values_numpy(
    df: pd.DataFrame,
    strategy: str = "median",
    target_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Imputes or drops missing values in columns.
    Supported strategies: 'median', 'mean', 'mode', or 'drop'.
    """
    cleaned_df = df.copy()
    imputation_details = {}
    
    if strategy == "drop":
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.dropna().reset_index(drop=True)
        imputation_details["_dropped_rows"] = {
            "imputed_count": initial_len - len(cleaned_df),
            "strategy": "drop",
            "replacement_value": "dropped"
        }
        return cleaned_df, imputation_details
        
    if target_columns is None:
        target_columns = [col for col in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[col])]
        
    for col in target_columns:
        if col not in cleaned_df.columns:
            continue
            
        arr = cleaned_df[col].to_numpy(dtype=np.float64, copy=True)
        nan_mask = np.isnan(arr)
        nan_count = int(np.sum(nan_mask))
        
        if nan_count > 0:
            valid_values = arr[~nan_mask]
            
            if len(valid_values) == 0:
                fill_val = 0.0
            elif strategy == "mean":
                fill_val = float(np.round(np.nanmean(arr), 2))
            elif strategy == "mode":
                s = cleaned_df[col].dropna()
                fill_val = float(s.mode()[0]) if not s.mode().empty else float(np.nanmedian(arr))
            else:  # Default to median
                fill_val = float(np.round(np.nanmedian(arr), 2))
                
            arr[nan_mask] = fill_val
            cleaned_df[col] = arr
            
            imputation_details[col] = {
                "imputed_count": nan_count,
                "strategy": strategy,
                "replacement_value": fill_val
            }
            
    # For categorical columns, impute with mode or 'Unknown'
    cat_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    for c_col in cat_cols:
        cat_nan_count = int(cleaned_df[c_col].isna().sum())
        if cat_nan_count > 0:
            mode_series = cleaned_df[c_col].dropna().mode()
            mode_val = str(mode_series[0]) if not mode_series.empty else "Unknown"
            cleaned_df[c_col] = cleaned_df[c_col].fillna(mode_val)
            imputation_details[c_col] = {
                "imputed_count": cat_nan_count,
                "strategy": "mode",
                "replacement_value": mode_val
            }
            
    return cleaned_df, imputation_details


def detect_iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects statistical outliers using the Interquartile Range (IQR) method.
    An observation is an outlier if x < Q1 - 1.5 * IQR or x > Q3 + 1.5 * IQR.
    """
    anomalies = []
    total_rows = len(df)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            arr = df[col].to_numpy(dtype=np.float64, copy=True)
            clean_arr = arr[~np.isnan(arr)]
            
            if len(clean_arr) >= 5:
                q1 = float(np.percentile(clean_arr, 25))
                q3 = float(np.percentile(clean_arr, 75))
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    out_mask = (clean_arr < lower_bound) | (clean_arr > upper_bound)
                    out_count = int(np.sum(out_mask))
                    
                    if out_count > 0:
                        anomalies.append({
                            "Column": col,
                            "Outlier Count": out_count,
                            "Outlier (%)": round((out_count / total_rows) * 100, 2),
                            "Q1 (25%)": round(q1, 2),
                            "Q3 (75%)": round(q3, 2),
                            "IQR": round(iqr, 2),
                            "Lower Bound": round(lower_bound, 2),
                            "Upper Bound": round(upper_bound, 2)
                        })
                        
    return pd.DataFrame(anomalies)


def handle_iqr_outliers(
    df: pd.DataFrame,
    strategy: str = "clip",
    target_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Handles statistical outliers detected via the IQR method.
    Strategy 'clip': Winsorizes values to IQR boundaries [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Strategy 'nan': Sets outliers to NaN for subsequent imputation.
    Strategy 'drop': Drops rows containing any outlier in target columns.
    """
    cleaned_df = df.copy()
    outlier_details = {}
    
    if target_columns is None:
        target_columns = [col for col in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[col])]
        
    if strategy == "drop":
        drop_mask = np.zeros(len(cleaned_df), dtype=bool)
        for col in target_columns:
            arr = cleaned_df[col].to_numpy(dtype=np.float64, copy=True)
            valid = ~np.isnan(arr)
            if np.sum(valid) >= 5:
                q1 = np.percentile(arr[valid], 25)
                q3 = np.percentile(arr[valid], 75)
                iqr = q3 - q1
                if iqr > 0:
                    out = valid & ((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr))
                    drop_mask |= out
                    outlier_details[col] = int(np.sum(out))
        cleaned_df = cleaned_df[~drop_mask].reset_index(drop=True)
        return cleaned_df, outlier_details

    for col in target_columns:
        if col not in cleaned_df.columns:
            continue
        arr = cleaned_df[col].to_numpy(dtype=np.float64, copy=True)
        valid = ~np.isnan(arr)
        
        if np.sum(valid) >= 5:
            q1 = np.percentile(arr[valid], 25)
            q3 = np.percentile(arr[valid], 75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                out_mask = valid & ((arr < lower) | (arr > upper))
                out_count = int(np.sum(out_mask))
                
                if out_count > 0:
                    if strategy == "clip":
                        arr[valid] = np.clip(arr[valid], lower, upper)
                        cleaned_df[col] = np.round(arr, 2)
                    elif strategy == "nan":
                        arr[out_mask] = np.nan
                        cleaned_df[col] = arr
                    outlier_details[col] = out_count
                    
    return cleaned_df, outlier_details


def detect_out_of_range(df: pd.DataFrame, custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    Detects physiological values outside standard biomedical limits using NumPy boolean comparisons.
    """
    bounds = custom_bounds or BIOMEDICAL_BOUNDS
    anomalies = []
    
    for col, (min_val, max_val) in bounds.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            arr = df[col].to_numpy(dtype=np.float64, copy=True)
            valid_mask = ~np.isnan(arr)
            out_mask = valid_mask & ((arr < min_val) | (arr > max_val))
            out_count = int(np.sum(out_mask))
            
            if out_count > 0:
                anomalies.append({
                    "Parameter": col,
                    "Expected Min": min_val,
                    "Expected Max": max_val,
                    "Observed Min": float(np.nanmin(arr)),
                    "Observed Max": float(np.nanmax(arr)),
                    "Out-of-Range Count": out_count
                })
                
    return pd.DataFrame(anomalies)


def clean_out_of_range(
    df: pd.DataFrame,
    strategy: str = "clip",
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Rectifies physiological out-of-range values using NumPy.
    Strategy 'clip': uses np.clip to bind values to plausible biomedical limits.
    """
    cleaned_df = df.copy()
    bounds = custom_bounds or BIOMEDICAL_BOUNDS
    clipped_details = {}
    
    for col, (min_val, max_val) in bounds.items():
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            arr = cleaned_df[col].to_numpy(dtype=np.float64, copy=True)
            valid_mask = ~np.isnan(arr)
            out_mask = valid_mask & ((arr < min_val) | (arr > max_val))
            out_count = int(np.sum(out_mask))
            
            if out_count > 0:
                if strategy == "clip":
                    arr[valid_mask] = np.clip(arr[valid_mask], min_val, max_val)
                    cleaned_df[col] = arr
                elif strategy == "nan":
                    arr[out_mask] = np.nan
                    cleaned_df[col] = arr
                    
                clipped_details[col] = out_count
                
    return cleaned_df, clipped_details


def detect_empty_and_constant_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detects empty columns (100% NaN) and constant columns (only 1 unique value).
    """
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and c not in empty_cols]
    return {
        "empty_columns": empty_cols,
        "constant_columns": constant_cols
    }


def calculate_bmi_numpy(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Vectorized BMI calculation from Height_cm and Weight_kg using NumPy.
    Formula: BMI = Weight_kg / (Height_m ^ 2)
    Safely prevents division by zero and negative heights using boolean indexing.
    """
    cleaned_df = df.copy()
    recalculated_count = 0
    
    if "Height_cm" in cleaned_df.columns and "Weight_kg" in cleaned_df.columns:
        heights = cleaned_df["Height_cm"].to_numpy(dtype=np.float64, copy=True)
        weights = cleaned_df["Weight_kg"].to_numpy(dtype=np.float64, copy=True)
        height_m = heights / 100.0
        
        valid_mask = (height_m >= 0.5) & (weights >= 10.0) & (~np.isnan(height_m)) & (~np.isnan(weights))
        calculated_bmi = np.full_like(weights, np.nan, dtype=np.float64)
        calculated_bmi[valid_mask] = np.round(weights[valid_mask] / (height_m[valid_mask] ** 2), 2)
        
        if "BMI" in cleaned_df.columns:
            existing_bmi = cleaned_df["BMI"].to_numpy(dtype=np.float64, copy=True)
            recalculated_count = int(np.sum(np.isnan(existing_bmi) & valid_mask))
            
        cleaned_df["BMI"] = calculated_bmi
        
    return cleaned_df, recalculated_count


def perform_full_cleaning_pipeline(
    df: pd.DataFrame,
    imputation_strategy: str = "median",
    clip_out_of_range: bool = True,
    handle_iqr: bool = False,
    iqr_strategy: str = "clip"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the comprehensive data-cleaning pipeline:
    1. Duplicate Removal
    2. Out-of-range detection & clipping
    3. Missing Value Imputation using NumPy
    4. Optional IQR statistical outlier treatment
    5. Vectorized BMI Calculation
    """
    audit = {
        "initial_rows": len(df),
        "initial_cols": len(df.columns),
        "duplicates_removed": 0,
        "imputation_strategy": imputation_strategy,
        "imputed_details": {},
        "clipped_details": {},
        "iqr_outliers_treated": {},
        "bmi_recalculated_count": 0,
        "final_rows": 0,
        "final_cols": 0
    }
    
    # Step 1: Remove duplicates
    step1_df, dup_count = remove_duplicates(df)
    audit["duplicates_removed"] = dup_count
    
    # Step 2: Handle physiological bounds
    if clip_out_of_range:
        step2_df, clipped_details = clean_out_of_range(step1_df, strategy="clip")
        audit["clipped_details"] = clipped_details
    else:
        step2_df = step1_df
        
    # Step 3: Handle missing values via NumPy
    step3_df, imputed_details = handle_missing_values_numpy(step2_df, strategy=imputation_strategy)
    audit["imputed_details"] = imputed_details
    
    # Step 4: Optional IQR Outlier treatment
    if handle_iqr:
        step4_df, iqr_details = handle_iqr_outliers(step3_df, strategy=iqr_strategy)
        audit["iqr_outliers_treated"] = iqr_details
    else:
        step4_df = step3_df
        
    # Step 5: Recalculate BMI consistently
    final_df, bmi_count = calculate_bmi_numpy(step4_df)
    audit["bmi_recalculated_count"] = bmi_count
    
    audit["final_rows"] = len(final_df)
    audit["final_cols"] = len(final_df.columns)
    
    return final_df, audit
