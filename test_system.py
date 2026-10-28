"""
=============================================================================
Test Suite: test_system.py
Health Data Analysis and Visualization System (MCA Minor Project)
=============================================================================
Tests all core functional requirements, edge cases, and enhanced modules.
"""

import unittest
import os
import io
import numpy as np
import pandas as pd

from modules.data_loader import (
    load_csv_data,
    load_default_sample_data,
    get_dataset_metadata,
    validate_health_columns
)
from modules.data_cleaning import (
    detect_missing_values,
    detect_duplicates,
    remove_duplicates,
    detect_out_of_range,
    clean_out_of_range,
    detect_iqr_outliers,
    handle_iqr_outliers,
    detect_empty_and_constant_columns,
    calculate_bmi_numpy,
    handle_missing_values_numpy,
    perform_full_cleaning_pipeline
)
from modules.statistical_analysis import (
    calculate_descriptive_stats_numpy,
    generate_dataset_summary_statistics,
    calculate_pearson_correlation_numpy,
    calculate_spearman_correlation_numpy,
    compute_correlation_matrix_numpy,
    categorize_age_groups,
    categorize_bmi,
    stratify_clinical_risks
)
from modules.insights import generate_automated_insights
from modules.dataset_comparison import compare_datasets
from modules.history_manager import (
    save_history_entry,
    load_history_entries,
    delete_history_entry,
    clear_all_history
)
from modules.report_generator import (
    generate_text_summary_report,
    generate_html_summary_report,
    generate_pdf_report
)


class TestHealthDataAnalysisSystem(unittest.TestCase):

    def setUp(self):
        """Set up standard valid test dataset and paths."""
        self.sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_health_data.csv")
        self.corrupted_path = os.path.join(os.path.dirname(__file__), "data", "test_corrupted_data.csv")

    # -----------------------------------------------------------------------
    # Test Case 1: Valid Dataset Loading
    # -----------------------------------------------------------------------
    def test_01_valid_dataset_loading(self):
        df, error = load_default_sample_data()
        self.assertIsNone(error, f"Expected no error, got: {error}")
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 100, "Dataset should contain at least 100 records as required.")
        self.assertIn("Patient_ID", df.columns)
        self.assertIn("Age", df.columns)
        self.assertIn("Systolic_BP", df.columns)
        
        meta = get_dataset_metadata(df)
        self.assertEqual(meta["total_records"], len(df))
        self.assertGreater(meta["numeric_columns_count"], 5)

    # -----------------------------------------------------------------------
    # Test Case 2: Missing Values Detection & NumPy Median Imputation
    # (Including Regression Test for StringDtype)
    # -----------------------------------------------------------------------
    def test_02_missing_values_detection_and_imputation(self):
        df, _ = load_default_sample_data()
        
        # Explicitly test with StringDtype column to prevent StringDtype issubdtype regression
        df_with_string_dtype = df.copy()
        if "Gender" in df_with_string_dtype.columns:
            df_with_string_dtype["Gender"] = df_with_string_dtype["Gender"].astype("string")
            
        missing_df = detect_missing_values(df_with_string_dtype)
        self.assertIsInstance(missing_df, pd.DataFrame)
        total_missing = missing_df["Missing Count"].sum()
        self.assertGreater(total_missing, 0, "Default sample data should contain intentional missing values.")
        
        cleaned_df, details = handle_missing_values_numpy(df_with_string_dtype, strategy="median")
        cleaned_missing_df = detect_missing_values(cleaned_df)
        remaining_missing = cleaned_missing_df["Missing Count"].sum()
        self.assertEqual(remaining_missing, 0, "After imputation, no missing values should remain in numeric columns.")

    # -----------------------------------------------------------------------
    # Test Case 3: Duplicate Record Detection and Removal
    # -----------------------------------------------------------------------
    def test_03_duplicate_records(self):
        df, _ = load_default_sample_data()
        dup_info = detect_duplicates(df)
        self.assertTrue(dup_info["has_duplicates"], "Sample data should have intentional duplicates.")
        
        initial_len = len(df)
        cleaned_df, removed_count = remove_duplicates(df)
        self.assertGreater(removed_count, 0)
        self.assertEqual(len(cleaned_df), initial_len - removed_count)
        
        post_dup_info = detect_duplicates(cleaned_df)
        self.assertFalse(post_dup_info["has_duplicates"], "No duplicates should remain after cleaning.")

    # -----------------------------------------------------------------------
    # Test Case 4: Incorrect Numerical Values & Out-of-Range Handling
    # -----------------------------------------------------------------------
    def test_04_out_of_range_values(self):
        df, _ = load_default_sample_data()
        anomalies = detect_out_of_range(df)
        self.assertFalse(anomalies.empty, "Sample data should contain intentional out-of-range values.")
        
        cleaned_df, clipped_details = clean_out_of_range(df, strategy="clip")
        post_anomalies = detect_out_of_range(cleaned_df)
        self.assertTrue(post_anomalies.empty, "After clipping, no physiological out-of-range values should remain.")

    # -----------------------------------------------------------------------
    # Test Case 5: Empty CSV File Handling
    # -----------------------------------------------------------------------
    def test_05_empty_csv_file(self):
        empty_buffer = io.StringIO("")
        df, error = load_csv_data(empty_buffer)
        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("empty", error.lower())

    # -----------------------------------------------------------------------
    # Test Case 6: Missing Optional Columns (Graceful Degradation)
    # -----------------------------------------------------------------------
    def test_06_missing_optional_columns(self):
        minimal_df = pd.DataFrame({
            "Patient_ID": ["P1", "P2", "P3"],
            "Age": [25, 45, 65],
            "Gender": ["Male", "Female", "Male"],
            "Blood_Glucose": [90, 110, 140]
        })
        
        cleaned_df, audit = perform_full_cleaning_pipeline(minimal_df)
        self.assertEqual(len(cleaned_df), 3)
        
        stats = generate_dataset_summary_statistics(cleaned_df)
        self.assertIn("Blood_Glucose", stats["Variable"].values)
        
        risks = stratify_clinical_risks(cleaned_df)
        self.assertIn("Glucose_Category", risks)
        self.assertNotIn("BP_Category", risks)

    # -----------------------------------------------------------------------
    # Test Case 7: Missing Required Columns / Non-health Dataset
    # -----------------------------------------------------------------------
    def test_07_missing_required_columns(self):
        random_df = pd.DataFrame({
            "City": ["Delhi", "Mumbai"],
            "Temperature": [32.5, 29.8]
        })
        val = validate_health_columns(random_df)
        self.assertFalse(val["is_valid"], "Non-health dataset should be flagged as lacking required health columns.")

    # -----------------------------------------------------------------------
    # Test Case 8: Invalid Height and Weight & Zero-Division BMI Protection
    # -----------------------------------------------------------------------
    def test_08_invalid_height_weight_and_bmi(self):
        test_df = pd.DataFrame({
            "Height_cm": [0.0, -10.0, 175.0, np.nan],
            "Weight_kg": [70.0, 65.0, 70.0, 80.0]
        })
        result_df, _ = calculate_bmi_numpy(test_df)
        bmis = result_df["BMI"].values
        
        self.assertTrue(np.isnan(bmis[0]), "Height = 0 must result in NaN, not ZeroDivisionError.")
        self.assertTrue(np.isnan(bmis[1]), "Negative height must result in NaN.")
        self.assertAlmostEqual(bmis[2], 22.86, places=1)

    # -----------------------------------------------------------------------
    # Test Case 9: Datasets with Constant Numerical Columns (Zero Variance)
    # -----------------------------------------------------------------------
    def test_09_constant_numerical_columns_correlation(self):
        test_df = pd.DataFrame({
            "Age": [30, 40, 50, 60],
            "Constant_BP": [120, 120, 120, 120],
            "Heart_Rate": [70, 75, 80, 85]
        })
        r, n_obs, msg = calculate_pearson_correlation_numpy(test_df["Age"].values, test_df["Constant_BP"].values)
        self.assertEqual(r, 0.0, "Correlation with constant column should be 0.0 without crashing.")
        
        corr_m, warnings = compute_correlation_matrix_numpy(test_df)
        self.assertNotIn("Constant_BP", corr_m.columns, "Constant column should be omitted from correlation matrix.")
        self.assertTrue(len(warnings) > 0)

    # -----------------------------------------------------------------------
    # Test Case 10: Dataset with No Usable Numerical Values
    # -----------------------------------------------------------------------
    def test_10_no_usable_numerical_values(self):
        text_df = pd.DataFrame({
            "Notes": ["Patient feeling well", "Routine checkup", "Follow-up"],
            "Location": ["Ward A", "Ward B", "Ward C"]
        })
        stats = generate_dataset_summary_statistics(text_df)
        self.assertTrue(stats.empty, "Descriptive statistics should be empty for non-numeric data.")
        
        corr_m, _ = compute_correlation_matrix_numpy(text_df)
        self.assertTrue(corr_m.empty, "Correlation matrix should be empty when no numeric columns exist.")

    # -----------------------------------------------------------------------
    # Test Case 11: Statistical IQR Outlier Detection & Treatment
    # -----------------------------------------------------------------------
    def test_11_iqr_outliers(self):
        # Create dataset with an obvious outlier
        normal_vals = list(range(100, 120))  # 20 points
        normal_vals.append(300)  # extreme outlier
        outlier_df = pd.DataFrame({"Test_BP": normal_vals})
        
        iqr_report = detect_iqr_outliers(outlier_df)
        self.assertFalse(iqr_report.empty, "IQR engine should detect the extreme outlier value 300.")
        self.assertEqual(int(iqr_report.iloc[0]["Outlier Count"]), 1)
        
        # Test clipping
        clipped_df, details = handle_iqr_outliers(outlier_df, strategy="clip")
        max_clipped = clipped_df["Test_BP"].max()
        self.assertLess(max_clipped, 300, "Clipped value should be bounded by upper IQR fence.")

    # -----------------------------------------------------------------------
    # Test Case 12: Spearman Rank Correlation
    # -----------------------------------------------------------------------
    def test_12_spearman_correlation(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        # Non-linear monotonic relationship: y = x^3
        y = x ** 3
        
        r_spearman, n_obs, _ = calculate_spearman_correlation_numpy(x, y)
        self.assertAlmostEqual(r_spearman, 1.0, places=2, msg="Spearman rank correlation of monotonic relation must be 1.0.")
        self.assertEqual(n_obs, 10)

    # -----------------------------------------------------------------------
    # Test Case 13: Automated Insight Generation
    # -----------------------------------------------------------------------
    def test_13_automated_insights(self):
        df, _ = load_default_sample_data()
        cleaned_df, audit = perform_full_cleaning_pipeline(df)
        insights = generate_automated_insights(cleaned_df, audit_log=audit)
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 2, "System should generate multiple analytical insights.")
        categories = [ins["category"] for ins in insights]
        self.assertIn("Data Quality", categories)

    # -----------------------------------------------------------------------
    # Test Case 14: Dataset Comparison Module
    # -----------------------------------------------------------------------
    def test_14_dataset_comparison(self):
        df_raw, _ = load_default_sample_data()
        df_clean, _ = perform_full_cleaning_pipeline(df_raw)
        
        comp = compare_datasets(df_raw, df_clean, label_a="Raw", label_b="Cleaned")
        self.assertEqual(comp["rows_a"], len(df_raw))
        self.assertEqual(comp["rows_b"], len(df_clean))
        self.assertIn("stats_comparison", comp)
        self.assertIsInstance(comp["stats_comparison"], pd.DataFrame)

    # -----------------------------------------------------------------------
    # Test Case 15: Local Analysis History Manager
    # -----------------------------------------------------------------------
    def test_15_history_manager(self):
        entry_id = save_history_entry(
            dataset_name="Test Dataset",
            record_count=100,
            column_count=12,
            audit_log={"duplicates_removed": 2, "imputed_details": {"Age": {"imputed_count": 3}}}
        )
        self.assertTrue(len(entry_id) > 0, "History entry ID should be generated.")
        
        entries = load_history_entries()
        self.assertGreater(len(entries), 0, "Loaded history should contain the saved entry.")
        
        deleted = delete_history_entry(entry_id)
        self.assertTrue(deleted, "Saved entry should be successfully deletable.")

    # -----------------------------------------------------------------------
    # Test Case 16: Multi-Format Report Generation
    # -----------------------------------------------------------------------
    def test_16_report_generation(self):
        df, _ = load_default_sample_data()
        cleaned_df, audit = perform_full_cleaning_pipeline(df)
        stats = generate_dataset_summary_statistics(cleaned_df)
        corr_m, _ = compute_correlation_matrix_numpy(cleaned_df)
        insights = generate_automated_insights(cleaned_df, audit_log=audit)
        
        # Text Report
        txt = generate_text_summary_report(cleaned_df, audit, stats, corr_m, insights=insights)
        self.assertIn("HEALTH DATA ANALYSIS", txt)
        self.assertIn("INSIGHTS", txt)
        
        # HTML Report
        html = generate_html_summary_report(cleaned_df, audit, stats, corr_m, insights=insights)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Health Data Analysis", html)
        
        # PDF Report
        pdf_bytes = generate_pdf_report(cleaned_df, audit, stats, corr_m, insights=insights)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"), "Generated PDF must have valid PDF-1.4 header.")
        self.assertTrue(pdf_bytes.rstrip().endswith(b"%%EOF"), "Generated PDF must conclude with valid %%EOF trailer.")
        self.assertGreater(len(pdf_bytes), 1000, "PDF byte length should be substantial.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
