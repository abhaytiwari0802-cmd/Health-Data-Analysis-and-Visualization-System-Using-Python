# Health Data Analysis and Visualization System Using Python
**MCA/BCA Minor Project | Master of Computer Applications**

A modular, robust, and interactive healthcare exploratory data analysis and visualization dashboard built with **Python 3**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **Streamlit**.

---

## 🌟 Key Features & Capabilities

1. **Executive Health Dashboard:**
   - Real-time KPI summary cards (Total Records, Numeric Fields, Missing Cells, Duplicate Rows, IQR Outliers, Average BMI).
   - Clinical risk stratification according to American Heart Association (AHA 2017) hypertension stages and American Diabetes Association (ADA) fasting glucose standards.
   - Demographic distribution charts (Age histogram with KDE, BMI classification, Gender breakdown).

2. **Smart Data Quality Analyzer & NumPy Cleaning Engine:**
   - Missing value detection compatible with all Pandas 2.x/3.x types (including ExtensionDtypes / StringDtype).
   - Median imputation (`np.nanmedian`), Mean, Mode, or Row-dropping.
   - Duplicate identification and removal (preserving first recorded entry).
   - Interquartile Range (IQR) statistical outlier detection ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$) and Winsorization (`np.clip`).
   - Physiological range validation against standard biomedical boundaries.
   - Vectorized BMI recalculation with division-by-zero protection.
   - Non-destructive processing: original raw dataset is preserved; cleaning is performed on an independent working copy with an audit log.

3. **Interactive Correlation Explorer:**
   - Supports both **Pearson correlation** ($r$, linear) and **Spearman rank correlation** ($\rho$, monotonic).
   - Annotated correlation matrix heatmaps with diverging colormaps.
   - Bivariate scatter explorer with linear trendlines ($y = mx + c$), observation counters ($N$), and statistical interpretations.
   - Zero-variance handling preventing division-by-zero errors.

4. **Automated Rule-Based Insight Generator:**
   - Derives evidence-based findings using actual cohort statistics (no external AI or API keys required).
   - Generates insights for dataset completeness, cleaning impact, distribution skewness, primary correlations, and clinical risk prevalence.

5. **Dataset Comparison Module:**
   - Compare "Raw Dataset vs. Cleaned Dataset" or upload a secondary CSV for side-by-side analysis.
   - Evaluates row/column deltas, missingness differences, duplicate count changes, and statistical parameter drift.

6. **Local Analysis History Manager:**
   - Saves lightweight analysis metadata snapshots to local JSON storage (`history/`).
   - Inspect, export, or delete previous run summaries without storing raw patient records.

7. **One-Click Multi-Format Export:**
   - Standalone multi-page binary PDF report (PDF 1.4) generated via pure Python byte streams without external C binary dependencies.
   - Cleaned dataset download (.csv).
   - Descriptive statistics table download (.csv).
   - Print-ready HTML medical audit report.

---

## 🏗️ Project Architecture

```
pythonproject/
│
├── app.py                     # Main interactive Streamlit dashboard
├── requirements.txt           # Project dependencies
├── test_system.py             # Automated unit test suite (16 test cases)
├── README.md                  # Project documentation and setup guide
│
├── data/                      # Dataset directory
│   ├── sample_health_data.csv # 163-record synthetic health benchmark dataset
│   ├── test_corrupted_data.csv# Corrupted test dataset for robustness verification
│   └── generate_data.py       # Benchmark dataset generator
│
├── modules/                   # Analytical business logic modules
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, validation, and schema synonym normalization
│   ├── data_cleaning.py       # NumPy cleaning, IQR outliers, median imputation, and BMI engine
│   ├── statistical_analysis.py# Descriptive statistics, Pearson & Spearman correlation
│   ├── visualization.py       # Headless Matplotlib & Seaborn visualizations
│   ├── insights.py            # Rule-based automated statistical insights engine
│   ├── dataset_comparison.py  # Dataset comparison and statistical drift analysis
│   ├── history_manager.py     # Local JSON analysis history manager
│   └── report_generator.py    # Text, HTML, and pure-Python binary PDF generator
│
├── history/                   # Local storage for saved analysis history snapshots
├── reports/                   # Destination folder for exported analysis reports
└── docs/                      # Academic project documentation and viva prep guides
    ├── PROJECT_REPORT.md      # Comprehensive 23-chapter project report
    ├── VIVA_QUESTIONS.md      # 25 viva voce questions and model answers
    └── DEMO_SCRIPT.md         # College presentation and demonstration walkthrough
```

---

## 🚀 Quickstart & Setup Instructions (Windows)

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed on your system.
- Standard Command Prompt (`cmd`) or PowerShell.

### 2. Setup Virtual Environment
Open terminal in the project directory:
```powershell
# Open project directory
cd c:\Users\abhay\Downloads\pythonproject

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment:
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Execute the test suite verifying all 16 test cases:
```powershell
python test_system.py
```
Expected output:
```
Ran 16 tests in 0.139s
OK
```

---

## ⚖️ Academic Disclaimer
This project is developed exclusively for educational analysis and academic evaluation as an MCA/BCA minor project. It demonstrates practical implementations of Python programming, NumPy vector computing, and exploratory data analysis.
