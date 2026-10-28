"""
Module: data_loader.py
Description: Handles loading, parsing, schema validation, and standardization
             of health datasets from CSV files or uploaded file streams.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Standard expected column names in health datasets
STANDARD_COLUMNS = [
    "Patient_ID",
    "Age",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Glucose",
    "Cholesterol",
    "Hemoglobin"
]

# Mapping dictionary to accommodate common column name variations
COLUMN_SYNONYMS = {
    "patient_id": "Patient_ID",
    "patientid": "Patient_ID",
    "id": "Patient_ID",
    "pid": "Patient_ID",
    "age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "height": "Height_cm",
    "height_cm": "Height_cm",
    "height(cm)": "Height_cm",
    "weight": "Weight_kg",
    "weight_kg": "Weight_kg",
    "weight(kg)": "Weight_kg",
    "bmi": "BMI",
    "systolic": "Systolic_BP",
    "systolic_bp": "Systolic_BP",
    "sys_bp": "Systolic_BP",
    "systolicbp": "Systolic_BP",
    "diastolic": "Diastolic_BP",
    "diastolic_bp": "Diastolic_BP",
    "dia_bp": "Diastolic_BP",
    "diastolicbp": "Diastolic_BP",
    "heart_rate": "Heart_Rate",
    "heartrate": "Heart_Rate",
    "pulse": "Heart_Rate",
    "pulse_rate": "Heart_Rate",
    "blood_glucose": "Blood_Glucose",
    "glucose": "Blood_Glucose",
    "bloodglucose": "Blood_Glucose",
    "sugar": "Blood_Glucose",
    "blood_sugar": "Blood_Glucose",
    "cholesterol": "Cholesterol",
    "chol": "Cholesterol",
    "serum_cholesterol": "Cholesterol",
    "hemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "hemo": "Hemoglobin"
}

NUMERIC_COLUMNS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Glucose",
    "Cholesterol",
    "Hemoglobin"
]


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes DataFrame column names by trimming whitespace,
    lowercasing for lookup against synonym dictionary, and returning canonical names.
    """
    new_cols = {}
    for col in df.columns:
        clean_key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if clean_key in COLUMN_SYNONYMS:
            new_cols[col] = COLUMN_SYNONYMS[clean_key]
        else:
            new_cols[col] = str(col).strip()
    return df.rename(columns=new_cols)


def load_csv_data(file_source: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads a dataset from a file path or file-like object (e.g. Streamlit UploadedFile).
    
    Parameters:
        file_source: Path string or BytesIO/StringIO buffer.
        
    Returns:
        Tuple of (pd.DataFrame, error_message).
        If successful, error_message is None; otherwise DataFrame is None.
    """
    try:
        if file_source is None:
            return None, "No file source provided."
        
        # Check file path existence if string
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                return None, f"File not found at path: {file_source}"
            if os.path.getsize(file_source) == 0:
                return None, "The specified CSV file is empty (0 bytes)."
        
        # Read CSV with pandas
        df = pd.read_csv(file_source)
        
        if df.empty:
            return None, "Dataset is empty: contains no records or rows."
        
        # Standardize column headers
        df = standardize_column_names(df)
        
        # Coerce known numeric columns safely (invalid strings become NaN for NumPy handling)
        for col in df.columns:
            if col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df, None

    except pd.errors.EmptyDataError:
        return None, "Validation Error: CSV file is empty."
    except pd.errors.ParserError as pe:
        return None, f"CSV Parsing Error: Malformed CSV content ({str(pe)})."
    except Exception as e:
        return None, f"Unexpected error loading dataset: {str(e)}"


def load_default_sample_data() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads the pre-packaged synthetic health dataset for immediate demonstration.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(base_dir, "data", "sample_health_data.csv")
    return load_csv_data(sample_path)


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates essential structural metadata for the dataset.
    """
    if df is None or df.empty:
        return {
            "total_records": 0,
            "total_columns": 0,
            "numeric_columns_count": 0,
            "categorical_columns_count": 0,
            "memory_usage_kb": 0.0,
            "column_info": []
        }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    memory_kb = df.memory_usage(deep=True).sum() / 1024.0
    
    col_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / len(df)) * 100, 2)
        unique_count = int(df[col].nunique())
        col_info.append({
            "Column Name": col,
            "Data Type": col_type,
            "Missing Values": null_count,
            "Missing (%)": null_pct,
            "Unique Values": unique_count
        })
        
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "numeric_columns_count": len(numeric_cols),
        "categorical_columns_count": len(categorical_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "memory_usage_kb": round(memory_kb, 2),
        "column_info": pd.DataFrame(col_info)
    }


def validate_health_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates presence of core clinical and demographic columns.
    """
    available_cols = set(df.columns)
    present_standard = [col for col in STANDARD_COLUMNS if col in available_cols]
    missing_standard = [col for col in STANDARD_COLUMNS if col not in available_cols]
    
    # Check minimum viable columns for any health analysis
    has_age = "Age" in available_cols
    has_vitals = any(c in available_cols for c in ["Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Glucose"])
    
    is_valid_health_dataset = has_age or has_vitals
    
    return {
        "is_valid": is_valid_health_dataset,
        "present_standard": present_standard,
        "missing_standard": missing_standard,
        "coverage_percentage": round((len(present_standard) / len(STANDARD_COLUMNS)) * 100, 1)
    }
