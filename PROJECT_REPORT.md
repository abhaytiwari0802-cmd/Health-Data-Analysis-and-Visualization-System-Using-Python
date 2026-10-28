# HEALTH DATA ANALYSIS AND VISUALIZATION SYSTEM USING PYTHON
**A Minor Project Report Submitted in Partial Fulfillment for the Degree of**  
**Master of Computer Applications (MCA)**

---

## 1. Project Title
**Health Data Analysis and Visualization System Using Python**

---

## 2. Abstract
With the rapid digitization of electronic health records (EHR) and biometric diagnostic monitoring, modern healthcare generates vast volumes of multidimensional physiological data. However, raw clinical datasets frequently suffer from data quality issues, including missing observations, duplicated patient profiles, recording entry errors, and out-of-range physiological measurements. 

This project presents the **Health Data Analysis and Visualization System Using Python**, an educational and exploratory data analysis (EDA) software platform designed to ingest, validate, clean, analyze, and visualize patient health records. The system utilizes **Python 3** as its foundation, harnessing **NumPy** for high-performance numerical array operations, missing-value imputation via robust statistical medians, outlier bounding via vector clipping, and mathematical Pearson correlation coefficient determination. **Pandas** is employed for schema normalization and tabular manipulation, while **Matplotlib** and **Seaborn** generate publication-grade statistical charts including KDE age distributions, box plots, WHO BMI classifications, and annotated correlation heatmaps. A reactive, modern web interface built on **Streamlit** provides multi-parameter filtering, dynamic visualization updates, and multi-format reporting (CSV, styled HTML, and standalone PDF). The project demonstrates an end-to-end data pipeline tailored for academic presentation and college-level viva evaluation.

---

## 3. Introduction
Exploratory Data Analysis (EDA) plays a foundational role in data science and computer science curricula. In healthcare informatics, extracting meaningful statistical patterns from patient measurements—such as blood pressure, heart rate, fasting blood glucose, and serum cholesterol—demands rigorous data preprocessing and numerical reliability.

The **Health Data Analysis and Visualization System** translates theoretical data science concepts into practical software engineering. Developed as an MCA minor project, the application bridges numerical computing with interactive dashboard visualization. The software guarantees reproducible data cleaning through explicit NumPy vectorized routines, provides transparent audit trails of every data transformation, and enables users to interrogate biomedical variables through dynamic visual interactions.

---

## 4. Problem Statement
Healthcare data acquired from clinical laboratories, wearable sensors, and institutional surveys often contains anomalies that hinder statistical inference:
1. **Missing Data:** Patients may fail to complete specific laboratory tests, leading to missing values (NaNs) that crash naive analytical scripts.
2. **Data Duplication:** Redundant patient entries skew cohort averages and distort demographic proportions.
3. **Extreme Outliers and Recording Errors:** Typing mistakes (such as negative heights, systolic blood pressure entries of 300+ mmHg, or resting heart rates of 0) invalidate statistical summaries.
4. **Lack of User-Friendly Tooling:** Many clinical researchers and academic students lack easy-to-use, localized desktop tools to clean datasets, compute descriptive metrics, and inspect variable correlations without writing ad-hoc scripts.
5. **Medical Misinterpretation Risks:** Inexperienced analysts frequently conflate statistical correlation with biological causality, necessitating clear educational guidance within analytical software.

---

## 5. Objectives
The key objectives of this project are:
1. **Develop a Modular Data Cleaning Pipeline:** Implement explicit NumPy-based routines for missing value detection, median imputation (`np.nanmedian`), outlier bounding (`np.clip`), and vectorized BMI calculation.
2. **Compute Descriptive Statistics:** Calculate comprehensive summary metrics (mean, median, standard deviation, variance, minimum, maximum, quartiles Q1/Q3, and interquartile range IQR) dynamically across all available numeric variables.
3. **Analyze Demographic Distributions:** Categorize patients into configurable age cohorts and WHO Body Mass Index classifications.
4. **Perform Correlation Analysis:** Implement Pearson correlation mathematically using NumPy array operations to evaluate linear associations between vital signs and biomarkers.
5. **Construct an Interactive Web Dashboard:** Provide an accessible, responsive user interface using Streamlit featuring 9 dedicated operational sections and real-time multi-attribute filtering.
6. **Support Multi-Format Reporting:** Enable instantaneous export of cleaned datasets (CSV), descriptive statistical tables (CSV), printable HTML medical audits, and standalone binary PDF reports.

---

## 6. Scope of the Project
- **Target Audience:** MCA students, data science researchers, biomedical informatics trainees, and academic mentors.
- **Data Scope:** Tabular patient datasets featuring demographic attributes (Age, Gender, Height, Weight, BMI) and vital signs/biomarkers (Blood Pressure, Heart Rate, Fasting Blood Glucose, Serum Cholesterol, Hemoglobin).
- **Operational Boundaries:** The application runs locally offline without external cloud dependencies, API subscriptions, or proprietary database engines.
- **Regulatory Clarification:** The system is explicitly bounded as an **educational analytical tool**. It does not output medical prescriptions, disease diagnostics, or automated treatment decisions.

---

## 7. Existing System and Proposed System

### 7.1 Existing System
* Relies on static spreadsheet software (e.g., Microsoft Excel) or one-off Python scripts.
* Missing values are often deleted entirely (listwise deletion), discarding valuable patient records and biasing samples.
* Lacks automated validation against physiological boundaries (e.g., negative heights or impossible heart rates go undetected).
* Code is typically monolithic (written in a single notebook or script), making it difficult to maintain, test, or defend in an MCA viva.
* Visualizations are static and require manual regeneration whenever filters change.

### 7.2 Proposed System
* **Modular Software Engineering:** Cleanly separated architecture across dedicated modules (`data_loader`, `data_cleaning`, `statistical_analysis`, `visualization`, and `report_generator`).
* **NumPy-Driven Imputation:** Preserves statistical distributions by substituting missing numerical values with robust column medians.
* **Vectorized Efficiency:** Utilizes NumPy boolean masking and vectorized array math for instantaneous data operations.
* **Reactive Web UI:** Streamlit interface provides live interactive sliders for age, blood pressure, and glucose, instantly updating all charts and summary metrics.
* **Automated Unit Testing:** Includes 11 automated unit tests verifying schema handling, edge cases, and report integrity.

---

## 8. Hardware and Software Requirements

### 8.1 Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher.
- **Memory (RAM):** Minimum 4 GB (8 GB recommended for large datasets).
- **Hard Disk Space:** Minimum 200 MB free disk space.
- **Display Resolution:** 1280 x 720 or higher.

### 8.2 Software Requirements
- **Operating System:** Windows 10/11, Linux (Ubuntu/Debian), or macOS.
- **Programming Language:** Python 3.10 to 3.13.
- **Core Libraries:**
  - `numpy` (>= 1.24.0) – Numerical array processing and vector math.
  - `pandas` (>= 2.0.0) – Tabular data management and CSV parsing.
  - `matplotlib` (>= 3.7.0) – Core visualization plotting.
  - `seaborn` (>= 0.12.0) – Advanced statistical visualizations and heatmaps.
  - `streamlit` (>= 1.25.0) – Web dashboard application framework.
- **Web Browser:** Google Chrome, Microsoft Edge, Mozilla Firefox, or Safari.

---

## 9. System Architecture

```
+-----------------------------------------------------------------------+
|                             USER INTERFACE                            |
|                 (Streamlit Web Application - app.py)                  |
|    - Sidebar Navigation (9 Sections)   - Real-time Cohort Filtering   |
|    - Interactive Sliders & Inputs      - Multi-format Export Triggers |
+------------------------------------+----------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                           APPLICATION LAYER                           |
+------------------------------------+----------------------------------+
| modules/data_loader.py             | modules/data_cleaning.py         |
| - CSV File Ingestion               | - Missing Value Imputation       |
| - Schema Validation & Synonym Map  | - Outlier Bounding (np.clip)     |
| - Metadata Extraction              | - Duplicate Record Elimination   |
|                                    | - Vectorized BMI Engine          |
+------------------------------------+----------------------------------+
| modules/statistical_analysis.py    | modules/visualization.py         |
| - NumPy Descriptive Statistics     | - Age & Gender Histograms        |
| - Pearson Correlation Formula      | - WHO BMI Category Charts        |
| - Demographic Age Grouping         | - Box Plots & Vital KDE Curves   |
| - Clinical Risk Stratification     | - Annotated Correlation Heatmaps |
+------------------------------------+----------------------------------+
| modules/report_generator.py                                           |
| - Plain-Text Academic Audit Report                                    |
| - Print-Ready HTML Medical Summary                                    |
| - Pure-Python Binary PDF 1.4 Generator                                |
+------------------------------------+----------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                              DATA LAYER                               |
|   data/sample_health_data.csv   |   Uploaded Custom Patient CSVs      |
+-----------------------------------------------------------------------+
```

---

## 10. Data Flow Diagram (DFD)

### Level 0 DFD (Context Level)
```
[ User / Student ] ---> ( Upload CSV / Set Filters ) ---> [ Health Data Analysis System ]
[ User / Student ] <--- ( Visualizations / Metrics / PDF ) <--- [ Health Data Analysis System ]
```

### Level 1 DFD (Process Level)
```
[ Raw CSV Input ]
       |
       v
( Process 1.0: Ingestion & Schema Standardization ) ---> [ Validated Schema ]
       |
       v
( Process 2.0: NumPy Data Cleaning & Imputation ) ------> [ Cleaned Data Store ]
       |                                                       |
       +-----------------------+-------------------------------+
                               |
       +-----------------------+-------------------------------+
       |                                                       |
       v                                                       v
( Process 3.0: Statistical & Correlation Engine )     ( Process 4.0: Visualization Engine )
       |                                                       |
       +-----------------------+-------------------------------+
                               |
                               v
( Process 5.0: Streamlit Reactive Dashboard & Multi-Format Report Exporter )
       |
       +---> [ Cleaned CSV ] | [ HTML Report ] | [ Binary PDF ] | [ Interactive UI ]
```

---

## 11. Module Descriptions

### 11.1 `modules/data_loader.py`
- Ingests CSV files from disk paths or browser-uploaded memory streams.
- Implements `COLUMN_SYNONYMS` dictionary mapping common clinical headers (e.g. `heartrate` $\to$ `Heart_Rate`, `systolic` $\to$ `Systolic_BP`).
- Converts numeric columns safely via `pd.to_numeric(errors='coerce')`, ensuring non-numeric corrupted strings become NaNs for downstream handling.
- Validates dataset schema completeness and calculates memory footprint.

### 11.2 `modules/data_cleaning.py`
- **Missing Value Detection:** Scans arrays using `np.isnan()` to locate missing values.
- **NumPy Median Imputation:** Converts columns to float arrays, calculates `np.nanmedian()`, and applies updates through boolean mask indexing (`arr[nan_mask] = median_val`).
- **Duplicate Removal:** Identifies and drops redundant patient records while preserving an audit trail of removed rows.
- **Biomedical Range Clipping:** Evaluates values against physiological reference boundaries and binds out-of-range anomalies via `np.clip()`.
- **Vectorized BMI Calculation:** Computes $BMI = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$ safely using boolean array indexing to prevent zero division.

### 11.3 `modules/statistical_analysis.py`
- Implements pure NumPy descriptive statistics: Mean (`np.mean`), Median (`np.median`), Variance (`np.var`), Standard Deviation (`np.std`), Min/Max, and Quartiles (`np.percentile`).
- Computes demographic age cohorts and WHO BMI categories.
- Implements the mathematical Pearson correlation formula using array difference products and square sums, with zero-variance safety checks.
- Categorizes clinical risks into Blood Pressure stages (AHA 2017) and Fasting Glucose categories (ADA).

### 11.4 `modules/visualization.py`
- Generates headless Matplotlib figures (`matplotlib.use('Agg')`) safe for multi-threaded web servers.
- Renders KDE distribution histograms with annotated vertical mean/median markers.
- Renders box plots for vital signs to highlight medians, quartiles, and outliers.
- Renders annotated correlation heatmaps with diverging `coolwarm` colormaps.
- Constructs scatter plots complete with NumPy linear regression trendlines ($y = mx + c$) and Pearson $r$ annotations.

### 11.5 `modules/report_generator.py`
- Builds a formatted plain-text academic audit report.
- Constructs a responsive, styled HTML medical audit report with KPI cards and tables, print-ready for browser-based PDF printing.
- Implements `SimplePDFWriter`, a pure-Python binary PDF 1.4 engine that outputs standard multi-page PDF documents without requiring external C libraries or third-party binaries.

---

## 12. Dataset Description and Sample Records

The pre-packaged synthetic health dataset contains 163 patient records constructed around physiological distributions.

| Attribute Name | Data Type | Measurement Unit | Clinical Description | Reference Range |
| :--- | :--- | :--- | :--- | :--- |
| `Patient_ID` | String | Identifier | Unique fictional alphanumeric patient ID | PID-1001 to PID-1160 |
| `Age` | Float/Int | Years | Age of the patient | 12 to 85 |
| `Gender` | String | Categorical | Biological sex (Male, Female, Other) | Male / Female / Other |
| `Height_cm` | Float | Centimeters (cm) | Standing height | 140.0 to 205.0 cm |
| `Weight_kg` | Float | Kilograms (kg) | Body weight | 42.0 to 130.0 kg |
| `BMI` | Float | $\text{kg/m}^2$ | Body Mass Index | 16.0 to 44.0 |
| `Systolic_BP` | Float/Int | mmHg | Peak arterial pressure during cardiac contraction | 90 to 195 mmHg |
| `Diastolic_BP` | Float/Int | mmHg | Resting arterial pressure between beats | 58 to 115 mmHg |
| `Heart_Rate` | Float/Int | beats/min (bpm) | Resting pulse rate | 55 to 115 bpm |
| `Blood_Glucose` | Float | mg/dL | Fasting blood glucose concentration | 68 to 290 mg/dL |
| `Cholesterol` | Float | mg/dL | Total serum cholesterol level | 125 to 330 mg/dL |
| `Hemoglobin` | Float | g/dL | Oxygen-carrying protein in red blood cells | 9.5 to 18.2 g/dL |

---

## 13. Algorithms and Methodology

### 13.1 Missing Value Imputation Algorithm
```
Algorithm: NumPy Median Imputation
Input: Numeric column array X of length N containing NaNs
Output: Clean array X_clean with NaNs replaced by column median

1. Extract X as a 64-bit float NumPy array.
2. Generate boolean mask: M = np.isnan(X)
3. Count NaNs: count = np.sum(M)
4. IF count == 0 THEN return X
5. Calculate median of valid observations:
      valid_elements = X[~M]
      IF len(valid_elements) == 0 THEN median_val = 0.0
      ELSE median_val = np.nanmedian(X)
6. Assign replacement value across mask:
      X[M] = median_val
7. Return X and audit metadata.
```

### 13.2 Vectorized BMI Calculation Algorithm
```
Algorithm: Safe Vectorized BMI Computation
Input: Height array H (in cm), Weight array W (in kg)
Output: BMI array B (in kg/m^2)

1. Convert height to meters: H_m = H / 100.0
2. Allocate output array B of shape matching W filled with np.nan.
3. Construct validity boolean mask:
      valid = (H_m >= 0.5) AND (W >= 10.0) AND (~np.isnan(H_m)) AND (~np.isnan(W))
4. Compute BMI only for valid elements:
      B[valid] = np.round(W[valid] / (H_m[valid] ** 2), 2)
5. Return B.
```

---

## 14. NumPy Operations Used in the Project
The project demonstrates extensive, explicit usage of NumPy functions:
1. `np.isnan(arr)` – Element-wise boolean detection of missing floating-point values.
2. `np.nanmedian(arr)` – Computes median value while ignoring NaN elements.
3. `np.nanmean(arr)` – Computes arithmetic mean while ignoring NaN elements.
4. `np.clip(arr, min_val, max_val)` – Constrains array values within physiological boundaries.
5. `np.mean()`, `np.std(ddof=1)`, `np.var(ddof=1)` – Calculates descriptive moments with Bessel's correction.
6. `np.percentile(arr, q)` – Calculates 25th (Q1) and 75th (Q3) percentiles for IQR calculation.
7. `np.sum()`, `np.sqrt()` – Vectorized sum of products and square roots for Pearson correlation.
8. `np.polyfit(x, y, deg=1)` – Least-squares polynomial fit for linear scatter trendlines.
9. `np.polyval(coeff, x_vals)` – Evaluates linear trendlines across dense continuous domains.
10. `np.full_like()` – Allocates typed arrays pre-filled with sentinels to avoid runtime division warnings.

---

## 15. Statistical Formulas Used

### 15.1 Descriptive Statistics
$$\bar{X} = \frac{1}{N} \sum_{i=1}^N X_i$$

$$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (X_i - \bar{X})^2}$$

$$\text{IQR} = Q_3 - Q_1$$

### 15.2 Pearson Correlation Coefficient ($r$)
$$r = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \cdot \sum_{i=1}^n (Y_i - \bar{Y})^2}}$$

---

## 16. Visualization Techniques
- **Kernel Density Estimation (KDE):** Smoothed continuous probability density curves superimposed on empirical histograms.
- **Box and Whisker Plots:** Displays 5-point summaries (minimum, Q1, median, Q3, maximum) and identifies outliers beyond $1.5 \times \text{IQR}$.
- **Annotated Heatmaps:** Visualizes multi-variable correlation matrices with numerical value overlays and diverging color gradients.
- **Bivariate Scatter Plots with Trendlines:** Shows individual patient observations colored by gender with superimposed linear regression lines.

---

## 17. Implementation Details
The application is structured into modular Python files under `modules/` and orchestrated via `app.py`. Streamlit session state (`st.session_state`) persists:
- `raw_df`: Original imported dataset.
- `cleaned_df`: Dataset post-cleaning.
- `cleaning_audit`: Dictionary recording all transformations.
- `data_source`: Description of active data origin.

---

## 18. Testing and Results
An automated test suite (`test_system.py`) validates system functionality across 11 test cases:
1. Valid dataset ingestion and metadata extraction.
2. Missing value detection and NumPy median imputation.
3. Duplicate record identification and removal.
4. Out-of-range value detection and bounding.
5. Empty CSV handling with user-friendly errors.
6. Graceful degradation when optional health columns are absent.
7. Validation of non-health datasets.
8. Invalid height/weight handling and zero-division protection in BMI calculation.
9. Constant numerical column handling (zero variance) in correlation analysis.
10. Datasets with no usable numeric columns.
11. Multi-format report generation (Text, HTML, PDF).

**Result:** All 11 automated test cases pass in under 0.1 seconds with 0 errors and 0 warnings.

---

## 19. Screenshots to Capture for Final Project Binder
During your college presentation, capture screenshots of the following views:
1. **Executive Home Dashboard:** KPI metric cards, AHA Blood Pressure stages, and ADA Glucose charts.
2. **Dataset Management & Schema View:** Upload interface and schema coverage table.
3. **Data Cleaning Pipeline:** Before/after missing value tables and cleaning audit metrics.
4. **Demographic Analysis:** Age distribution histogram and configurable age cohort sliders.
5. **Vital Signs Analysis:** Box plots and 9-point descriptive statistics table.
6. **Correlation Analysis:** Annotated correlation heatmap and interactive scatter plot with trendline.
7. **Patient Records Browser:** Filtered records table and search query filter.
8. **Export & Reports:** Download buttons and on-screen analytical preview.
9. **Project Architecture View:** About Project screen displaying formulas and tech stack.

---

## 20. Advantages and Limitations

### 20.1 Advantages
- **Robust Cleaning:** Preserves data volume through median imputation rather than record deletion.
- **High Performance:** Vectorized NumPy operations execute in milliseconds.
- **Zero Heavy Dependencies:** Runs locally without external database setup or cloud subscriptions.
- **Dynamic Adaptability:** Gracefully analyzes datasets even if optional vital signs are absent.
- **Educational Value:** Fully documented codebase ideal for viva examination.

### 20.2 Limitations
- Evaluates linear relationships via Pearson correlation; nonlinear interactions require Spearman rank or machine learning models.
- Intended for tabular batch CSV files rather than real-time streaming hospital telemetry.
- Educational tool; cannot be utilized for autonomous clinical diagnosis.

---

## 21. Future Enhancements
- Integration with SQL databases (SQLite / PostgreSQL) for persistent multi-user storage.
- Implementation of supervised machine learning classifiers (Logistic Regression, Random Forest) for disease risk prediction.
- Support for longitudinal patient tracking across multiple follow-up visits.
- REST API endpoints using FastAPI for mobile integration.

---

## 22. Conclusion
The **Health Data Analysis and Visualization System Using Python** demonstrates the practical application of core data science and software engineering principles. By uniting Python, NumPy, Pandas, Matplotlib, Seaborn, and Streamlit, the system delivers an end-to-end analytical platform that addresses real-world clinical data challenges including missing values, duplicates, and out-of-range readings. The project successfully fulfills all academic requirements for an MCA minor project submission.

---

## 23. References
1. McKinney, W. (2022). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter* (3rd ed.). O'Reilly Media.
2. Harris, C. R., et al. (2020). "Array programming with NumPy." *Nature*, 585(7825), 357–362.
3. Waskom, M. L. (2021). "Seaborn: statistical data visualization." *Journal of Open Source Software*, 6(60), 3021.
4. American Heart Association (AHA). (2017). *Guidelines for High Blood Pressure in Adults*.
5. American Diabetes Association (ADA). (2023). *Standards of Care in Diabetes*.
6. World Health Organization (WHO). *Body Mass Index - BMI Classification Guidelines*.
