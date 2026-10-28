"""
Dataset Generator for Health Data Analysis System
Generates synthetic, clinically plausible patient records for MCA Minor Project testing and demonstration.
"""

import numpy as np
import pandas as pd
import os

def generate_sample_dataset(output_path="data/sample_health_data.csv", n_records=160, seed=42):
    np.random.seed(seed)
    
    # 1. Demographics
    patient_ids = [f"PID-{1000 + i + 1}" for i in range(n_records)]
    
    # Age: mix of young, middle-aged, and elderly
    ages = np.random.choice(
        np.concatenate([
            np.random.randint(12, 18, size=15),   # Pediatric/adolescent
            np.random.randint(18, 36, size=50),   # Young adults
            np.random.randint(36, 51, size=45),   # Middle age
            np.random.randint(51, 66, size=35),   # Older adults
            np.random.randint(66, 85, size=15)    # Seniors
        ]),
        size=n_records,
        replace=False
    )
    
    genders = np.random.choice(["Male", "Female", "Other"], size=n_records, p=[0.49, 0.48, 0.03])
    
    # Heights in cm (normally distributed around 170 for males, 160 for females)
    heights = np.zeros(n_records)
    for i, g in enumerate(genders):
        if g == "Male":
            heights[i] = np.round(np.random.normal(172, 7.5), 1)
        elif g == "Female":
            heights[i] = np.round(np.random.normal(160, 6.8), 1)
        else:
            heights[i] = np.round(np.random.normal(166, 7.0), 1)
            
    # Clip heights to realistic bounds
    heights = np.clip(heights, 140.0, 205.0)
    
    # Weights in kg (correlated with height + noise)
    # BMI = weight / (height/100)^2 -> weight = BMI * (height/100)^2
    target_bmis = np.random.normal(25.5, 4.8, size=n_records)
    weights = np.round(target_bmis * ((heights / 100.0) ** 2), 1)
    weights = np.clip(weights, 42.0, 130.0)
    
    # Actual initial BMI
    bmis = np.round(weights / ((heights / 100.0) ** 2), 2)
    
    # 2. Vitals & Clinical Metrics
    # Systolic BP: weakly correlated with age and BMI
    base_sys = 105 + (ages * 0.4) + (bmis * 0.6) + np.random.normal(0, 10, size=n_records)
    systolic_bp = np.round(np.clip(base_sys, 90, 195)).astype(int)
    
    # Diastolic BP: correlated with systolic BP (typically sys * 0.65 + noise)
    base_dia = (systolic_bp * 0.62) + np.random.normal(0, 5, size=n_records)
    diastolic_bp = np.round(np.clip(base_dia, 58, 115)).astype(int)
    
    # Heart Rate (bpm): normal resting between 60-100 bpm
    heart_rates = np.round(np.random.normal(74, 9, size=n_records)).astype(int)
    heart_rates = np.clip(heart_rates, 55, 115)
    
    # Blood Glucose (mg/dL): correlated with BMI and Age
    base_glucose = 80 + (bmis * 1.2) + (ages * 0.3) + np.random.exponential(scale=18, size=n_records)
    blood_glucose = np.round(np.clip(base_glucose, 68, 290), 1)
    
    # Cholesterol (mg/dL): correlated with Age and BMI
    base_chol = 150 + (ages * 0.9) + (bmis * 1.5) + np.random.normal(0, 22, size=n_records)
    cholesterol = np.round(np.clip(base_chol, 125, 330), 1)
    
    # Hemoglobin (g/dL): males ~13.5-17.5, females ~12.0-15.5
    hemoglobins = np.zeros(n_records)
    for i, g in enumerate(genders):
        if g == "Male":
            hemoglobins[i] = np.round(np.random.normal(15.0, 1.2), 1)
        else:
            hemoglobins[i] = np.round(np.random.normal(13.5, 1.1), 1)
    hemoglobins = np.clip(hemoglobins, 9.5, 18.2)
    
    # Build initial DataFrame
    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": ages,
        "Gender": genders,
        "Height_cm": heights,
        "Weight_kg": weights,
        "BMI": bmis,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Heart_Rate": heart_rates,
        "Blood_Glucose": blood_glucose,
        "Cholesterol": cholesterol,
        "Hemoglobin": hemoglobins
    })
    
    # INTENTIONAL IMPERFECTIONS FOR VIVA DEMONSTRATION:
    # 1. Missing Values (NaN) in a few columns
    df.loc[7, "Blood_Glucose"] = np.nan
    df.loc[19, "Cholesterol"] = np.nan
    df.loc[34, "Heart_Rate"] = np.nan
    df.loc[52, "Systolic_BP"] = np.nan
    df.loc[68, "Weight_kg"] = np.nan
    df.loc[68, "BMI"] = np.nan
    df.loc[88, "Hemoglobin"] = np.nan
    df.loc[105, "Height_cm"] = np.nan
    df.loc[105, "BMI"] = np.nan
    df.loc[122, "Blood_Glucose"] = np.nan
    
    # 2. Out-of-range / anomalous entries
    df.loc[15, "Systolic_BP"] = 320    # Clinically dangerous / entry error
    df.loc[42, "Heart_Rate"] = 240     # Out of plausible resting range
    df.loc[77, "Height_cm"] = 0        # Corrupted entry (0 cm)
    df.loc[77, "BMI"] = 0              # Invalid BMI
    
    # 3. Duplicate Records (exact copies of existing rows to test duplicate removal)
    duplicate_rows = df.iloc[[10, 25, 40]].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample health dataset generated at {output_path} with {len(df)} records.")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
