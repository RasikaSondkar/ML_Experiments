# Integrated Machine Learning Algorithm Dashboard
**College Mini Project — Practical Laboratory Consolidation**

---

## 1. Project Overview
The **Integrated Machine Learning Algorithm Dashboard** is a practical web application that consolidates 10 machine learning laboratory experiments into a single, cohesive, locally runnable interface. 

Rather than simulating predictions or hardcoding mock values, this application connects directly to pre-trained Scikit-learn pipelines and serialized models (`.pkl` and `.joblib`) created during laboratory sessions, evaluating them against genuine experimental datasets.

---

## 2. Integrated Algorithms Directory

| # | Algorithm Name | Category | Model File | Dataset | Key Metrics |
|---|---|---|---|---|---|
| 1 | **Simple Linear Regression** | Regression | `models/slr_model.pkl` | `Placement.csv` (200 records) | $R^2 = 0.7752$, MAE, RMSE |
| 2 | **Multiple Linear Regression** | Regression | `models/mlr_model.pkl` | `Cgpa_iq_package.csv` (200 records) | $R^2 = 0.8673$, MAE, Residuals |
| 3 | **Binary Logistic Regression** | Classification | `models/bc_model.pkl` | `Placement_Y_N.csv` (100 records) | Accuracy = 91.0%, F1-Score |
| 4 | **Multiclass Logistic Regression** | Classification | `models/mc_model.pkl` | `iris.csv` (150 records) | Accuracy = 98.0%, 3×3 Confusion Matrix |
| 5 | **Decision Tree Classifier** | Classification | `models/dt_model.pkl` | `heart_disease_dataset.csv` (303 records) | Accuracy = 85.15%, Gini Importance |
| 6 | **K-Nearest Neighbors (KNN)** | Classification | `models/knn_diabetes_model.joblib` | `diabetes_dataset.csv` (768 records) | Accuracy = 72.66%, Distance Metric |
| 7 | **Naive Bayes Classifier** | NLP | `models/nb_model.pkl` + `nb_vectorizer.pkl` | `college_messages.csv` (229 records) | Text Accuracy = 99.0%, Bayes Posterior |
| 8 | **Linear SVM** | Classification | `models/linear_svm_model.pkl` | `data_banknote_authentication.csv` (1,372 records) | Accuracy = 98.40%, Optimal Hyperplane |
| 9 | **Non-Linear SVM (RBF Kernel)** | Classification | `models/nonlinear_svm_model.pkl` | `iris.csv` (150 records) | Accuracy = 97.33%, Kernel Space Margin |
| 10 | **Random Forest Classifier** | Ensemble | `models/rf_classifier.joblib` | `diabetes.csv` (768 records) | Accuracy = 94.79%, 300 Bagged Trees |

---

## 3. Technology Stack
- **Backend**: Python 3, Flask
- **Machine Learning**: Scikit-learn, NumPy, Pandas, Joblib, Pickle
- **Visualizations**: Matplotlib, Seaborn (dynamically rendered as embedded Base64 images)
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript (Academic Minimalist Design)

---

## 4. Project Directory Structure
```
ml_algorithm_dashboard/
├── app.py                     # Flask application routes and controllers
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation and execution guide
├── models/                    # Trained model and pipeline files (.pkl, .joblib)
│   ├── slr_model.pkl
│   ├── mlr_model.pkl
│   ├── bc_model.pkl
│   ├── mc_model.pkl
│   ├── dt_model.pkl
│   ├── knn_diabetes_model.joblib
│   ├── nb_model.pkl
│   ├── nb_vectorizer.pkl
│   ├── linear_svm_model.pkl
│   ├── nonlinear_svm_model.pkl
│   └── rf_classifier.joblib
├── data/                      # Evaluation datasets (.csv)
│   ├── Placement.csv
│   ├── Cgpa_iq_package.csv
│   ├── Placement_Y_N.csv
│   ├── iris.csv
│   ├── heart_disease_dataset.csv
│   ├── diabetes_dataset.csv
│   ├── college_messages.csv
│   ├── data_banknote_authentication.csv
│   └── diabetes.csv
├── utils/
│   ├── __init__.py
│   ├── model_loader.py        # Model loading, caching, feature definitions, and inference
│   ├── evaluator.py           # Evaluation metrics computation directly on datasets
│   └── visualizer.py          # Dynamic Matplotlib chart generation (Base64 PNG)
├── static/
│   ├── css/
│   │   └── style.css          # Restrained academic styling (muted blue, off-white background)
│   └── js/
│       └── main.js            # Sample data prefill and UI utilities
└── templates/
    ├── base.html              # Fixed sidebar and global layout
    ├── index.html             # Dashboard home with algorithm cards and summary table
    ├── algorithm.html         # Individual algorithm workspace (form, metrics, chart, data preview)
    └── about.html             # Project details and viva reference guide
```

---

## 5. Setup & Running the Application

### Step 1: Open Terminal in the Project Directory
```powershell
cd C:\Users\rasik\.gemini\antigravity\scratch\ml_algorithm_dashboard
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Flask Web Application
```powershell
python app.py
```

### Step 4: Open in Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 6. Key Features for Examiner Evaluation
1. **Interactive Model Inference**: Test each model by entering individual feature values or clicking the **"Load Sample Values"** button to populate verified test samples with a single click.
2. **Actual Model Evaluation Metrics**: Every algorithm page displays true computed performance metrics ($R^2$, MAE, RMSE, Accuracy, Precision, Recall, F1-Score) calculated over the actual training datasets.
3. **Embedded Visualizations**: Matplotlib and Seaborn generate clean, high-resolution plots (regression lines, actual vs. predicted scatters, 2D decision boundaries, confusion matrices, and feature importance bar charts) that highlight the user's specific test point.
4. **Dataset Preview**: Inspect the top 5 records of each algorithm's corresponding training dataset directly in the browser.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e60f65ac-564e-482c-8a90-09b269c5dd20" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/316d79ae-0cb1-4325-9bd0-cf03537625c1" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a72052c9-5e91-4335-af25-61e8bc1f15fa" />



