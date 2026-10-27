import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

_LOADED_MODELS: Dict[str, Any] = {}

ALGORITHM_CONFIGS = {
    "simple-linear-regression": {
        "id": "simple-linear-regression",
        "name": "Simple Linear Regression",
        "category": "Regression",
        "lab_experiment": "ML Lab Experiment 01 / 02: Simple Linear Regression",
        "model_file": "slr_model.pkl",
        "dataset_file": "Placement.csv",
        "dataset_name": "College Student Placement Dataset (CGPA vs Package)",
        "summary": "Models the linear relationship between a single independent predictor (Student CGPA) and a continuous target (Salary Package in LPA) using Ordinary Least Squares (OLS) regression.",
        "theory": "Simple Linear Regression fits a straight line minimizing the sum of squared differences between observed and predicted values: Package = β₀ + β₁·(CGPA).",
        "target_type": "continuous",
        "target_label": "Predicted Package",
        "target_unit": "LPA",
        "features": [
            {
                "name": "cgpa",
                "label": "Student CGPA",
                "type": "number",
                "step": "0.01",
                "min": "0.0",
                "max": "10.0",
                "placeholder": "e.g. 7.85",
                "default": "7.85",
                "description": "Cumulative Grade Point Average (0.0 to 10.0)"
            }
        ],
        "sample_inputs": {"cgpa": 8.25}
    },
    "multiple-linear-regression": {
        "id": "multiple-linear-regression",
        "name": "Multiple Linear Regression",
        "category": "Regression",
        "lab_experiment": "ML Lab Experiment 02.1 / 03: Multivariate Linear Regression",
        "model_file": "mlr_model.pkl",
        "dataset_file": "Cgpa_iq_package.csv",
        "dataset_name": "Student CGPA & IQ vs Placement Package Dataset",
        "summary": "Extends linear regression to model a continuous dependent variable (Package) from multiple independent predictors (CGPA and IQ score).",
        "theory": "Multiple Linear Regression estimates multiple partial regression coefficients: Package = β₀ + β₁·(CGPA) + β₂·(IQ), capturing concurrent effects of academic and cognitive scores.",
        "target_type": "continuous",
        "target_label": "Predicted Package",
        "target_unit": "LPA",
        "features": [
            {
                "name": "cgpa",
                "label": "Student CGPA",
                "type": "number",
                "step": "0.01",
                "min": "0.0",
                "max": "10.0",
                "placeholder": "e.g. 7.90",
                "default": "7.90",
                "description": "Academic performance on a 10-point scale"
            },
            {
                "name": "iq",
                "label": "IQ Score",
                "type": "number",
                "step": "1",
                "min": "50",
                "max": "200",
                "placeholder": "e.g. 115",
                "default": "115",
                "description": "Standardized cognitive aptitude score"
            }
        ],
        "sample_inputs": {"cgpa": 8.10, "iq": 120}
    },
    "binary-logistic-regression": {
        "id": "binary-logistic-regression",
        "name": "Binary Logistic Regression",
        "category": "Classification (Binary)",
        "lab_experiment": "ML Lab Experiment 03 / 04: Binary Logistic Regression",
        "model_file": "bc_model.pkl",
        "dataset_file": "Placement_Y_N.csv",
        "dataset_name": "Campus Placement Status Dataset (CGPA & IQ)",
        "summary": "Classifies whether a student will secure campus placement using the sigmoid activation function mapped to binary probabilities (Placed vs Not Placed).",
        "theory": "Uses the logistic sigmoid function P(y=1) = 1 / (1 + e^-(z)) with standardized CGPA and IQ to establish an optimal linear decision boundary.",
        "target_type": "categorical",
        "target_label": "Placement Status",
        "features": [
            {
                "name": "cgpa",
                "label": "Student CGPA",
                "type": "number",
                "step": "0.01",
                "min": "0.0",
                "max": "10.0",
                "placeholder": "e.g. 7.50",
                "default": "7.50",
                "description": "Cumulative GPA of candidate"
            },
            {
                "name": "iq",
                "label": "IQ Score",
                "type": "number",
                "step": "1",
                "min": "50",
                "max": "200",
                "placeholder": "e.g. 112",
                "default": "112",
                "description": "Candidate aptitude benchmark"
            }
        ],
        "sample_inputs": {"cgpa": 7.80, "iq": 122}
    },
    "multiclass-logistic-regression": {
        "id": "multiclass-logistic-regression",
        "name": "Multiclass Logistic Regression",
        "category": "Classification (Multiclass)",
        "lab_experiment": "ML Lab Experiment 04 / 05: Multiclass Logistic Regression",
        "model_file": "mc_model.pkl",
        "dataset_file": "iris.csv",
        "dataset_name": "Fisher's Iris Botanical Dataset (150 Samples)",
        "summary": "Classifies botanical samples into three Iris flower species (Setosa, Versicolor, Virginica) using multinomial logistic regression.",
        "theory": "Applies the Softmax function over four morphological measurements: P(y=k|x) = exp(w_k·x) / Σ exp(w_j·x) to compute probabilities across three mutually exclusive species.",
        "target_type": "categorical",
        "target_label": "Predicted Iris Species",
        "features": [
            {
                "name": "sepal_length",
                "label": "Sepal Length (cm)",
                "type": "number",
                "step": "0.1",
                "min": "0.1",
                "max": "15.0",
                "placeholder": "e.g. 5.1",
                "default": "5.8",
                "description": "Length of outer floral calyx"
            },
            {
                "name": "sepal_width",
                "label": "Sepal Width (cm)",
                "type": "number",
                "step": "0.1",
                "min": "0.1",
                "max": "10.0",
                "placeholder": "e.g. 3.5",
                "default": "3.0",
                "description": "Width of outer floral calyx"
            },
            {
                "name": "petal_length",
                "label": "Petal Length (cm)",
                "type": "number",
                "step": "0.1",
                "min": "0.1",
                "max": "15.0",
                "placeholder": "e.g. 1.4",
                "default": "4.2",
                "description": "Length of inner flower petal"
            },
            {
                "name": "petal_width",
                "label": "Petal Width (cm)",
                "type": "number",
                "step": "0.1",
                "min": "0.1",
                "max": "10.0",
                "placeholder": "e.g. 0.2",
                "default": "1.3",
                "description": "Width of inner flower petal"
            }
        ],
        "sample_inputs": {"sepal_length": 6.4, "sepal_width": 2.8, "petal_length": 5.6, "petal_width": 2.1}
    },
    "decision-tree": {
        "id": "decision-tree",
        "name": "Decision Tree Classifier",
        "category": "Classification (Binary)",
        "lab_experiment": "ML Lab Experiment 06: Decision Tree with Gini Impurity",
        "model_file": "dt_model.pkl",
        "dataset_file": "heart_disease_dataset.csv",
        "dataset_name": "Cleveland Heart Disease Clinical Dataset (303 Records)",
        "summary": "Non-parametric supervised model predicting presence or absence of cardiovascular heart disease using recursive hierarchical splits guided by Gini impurity.",
        "theory": "Recursively partitions the 13 clinical feature dimensions to maximize Information Gain by minimizing Gini impurity: Gini = 1 - Σ (p_i)², providing clear rule-based interpretability.",
        "target_type": "categorical",
        "target_label": "Heart Disease Diagnosis",
        "features": [
            {"name": "age", "label": "Age (years)", "type": "number", "step": "1", "min": "18", "max": "100", "placeholder": "54", "default": "54", "description": "Patient age in years"},
            {"name": "sex", "label": "Sex", "type": "select", "options": [{"value": "1", "label": "1: Male"}, {"value": "0", "label": "0: Female"}], "default": "1", "description": "Biological sex"},
            {"name": "cp", "label": "Chest Pain Type", "type": "select", "options": [{"value": "0", "label": "0: Typical Angina"}, {"value": "1", "label": "1: Atypical Angina"}, {"value": "2", "label": "2: Non-anginal Pain"}, {"value": "3", "label": "3: Asymptomatic"}], "default": "0", "description": "Subjective thoracic discomfort pattern"},
            {"name": "trestbps", "label": "Resting Blood Pressure", "type": "number", "step": "1", "min": "60", "max": "240", "placeholder": "130", "default": "130", "description": "Resting blood pressure in mm Hg upon hospital admission"},
            {"name": "chol", "label": "Serum Cholesterol", "type": "number", "step": "1", "min": "100", "max": "600", "placeholder": "246", "default": "246", "description": "Total serum cholesterol in mg/dl"},
            {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl", "type": "select", "options": [{"value": "0", "label": "0: False (<= 120)"}, {"value": "1", "label": "1: True (> 120)"}], "default": "0", "description": "Glycemic metabolic biomarker"},
            {"name": "restecg", "label": "Resting ECG Results", "type": "select", "options": [{"value": "0", "label": "0: Normal"}, {"value": "1", "label": "1: ST-T Wave Abnormality"}, {"value": "2", "label": "2: Left Ventricular Hypertrophy"}], "default": "0", "description": "Resting electrocardiographic findings"},
            {"name": "thalach", "label": "Max Heart Rate Achieved", "type": "number", "step": "1", "min": "50", "max": "250", "placeholder": "150", "default": "150", "description": "Peak cardiac heart rate during stress exercise"},
            {"name": "exang", "label": "Exercise Induced Angina", "type": "select", "options": [{"value": "0", "label": "0: No"}, {"value": "1", "label": "1: Yes"}], "default": "0", "description": "Ischemia induced by physical exertion"},
            {"name": "oldpeak", "label": "ST Depression (Oldpeak)", "type": "number", "step": "0.1", "min": "0.0", "max": "10.0", "placeholder": "1.2", "default": "1.2", "description": "ST depression induced by exercise relative to rest"},
            {"name": "slope", "label": "Slope of Peak Exercise ST", "type": "select", "options": [{"value": "0", "label": "0: Upsloping"}, {"value": "1", "label": "1: Flat"}, {"value": "2", "label": "2: Downsloping"}], "default": "1", "description": "Electrocardiographic segment trajectory"},
            {"name": "ca", "label": "Major Vessels Colored by Fluoroscopy", "type": "select", "options": [{"value": "0", "label": "0 vessels"}, {"value": "1", "label": "1 vessel"}, {"value": "2", "label": "2 vessels"}, {"value": "3", "label": "3 vessels"}], "default": "0", "description": "Number of patent coronary arteries"},
            {"name": "thal", "label": "Thalassemia Scan", "type": "select", "options": [{"value": "1", "label": "1: Normal"}, {"value": "2", "label": "2: Fixed Defect"}, {"value": "3", "label": "3: Reversible Defect"}], "default": "2", "description": "Nuclear thallium stress perfusion test result"}
        ],
        "sample_inputs": {
            "age": 57, "sex": 1, "cp": 0, "trestbps": 140, "chol": 241,
            "fbs": 0, "restecg": 1, "thalach": 123, "exang": 1, "oldpeak": 0.2,
            "slope": 1, "ca": 0, "thal": 3
        }
    },
    "knn": {
        "id": "knn",
        "name": "K-Nearest Neighbors (KNN)",
        "category": "Classification (Binary)",
        "lab_experiment": "ML Lab Experiment 07: K-Nearest Neighbors Classifier",
        "model_file": "knn_diabetes_model.joblib",
        "dataset_file": "diabetes_dataset.csv",
        "dataset_name": "Pima Indians Diabetes Diagnostic Dataset (768 Patients)",
        "summary": "Instance-based non-parametric classifier predicting diabetes likelihood by measuring Euclidean distance across standardized biometric measurements to the top K nearest records.",
        "theory": "Calculates Euclidean distance d(x, x_i) = sqrt(Σ (x_j - x_ij)²) across all training samples after standard scaling, assigning class label by majority vote among the K=3 nearest neighbors.",
        "target_type": "categorical",
        "target_label": "Diabetes Outcome",
        "features": [
            {"name": "Pregnancies", "label": "Number of Pregnancies", "type": "number", "step": "1", "min": "0", "max": "20", "placeholder": "2", "default": "2", "description": "Number of times pregnant"},
            {"name": "Glucose", "label": "Plasma Glucose Concentration", "type": "number", "step": "1", "min": "0", "max": "300", "placeholder": "138", "default": "138", "description": "2-hour oral glucose tolerance test (mg/dl)"},
            {"name": "BloodPressure", "label": "Diastolic Blood Pressure", "type": "number", "step": "1", "min": "0", "max": "150", "placeholder": "72", "default": "72", "description": "Diastolic arterial pressure (mm Hg)"},
            {"name": "SkinThickness", "label": "Triceps Skinfold Thickness", "type": "number", "step": "1", "min": "0", "max": "100", "placeholder": "35", "default": "35", "description": "Subcutaneous adipose thickness (mm)"},
            {"name": "Insulin", "label": "2-Hour Serum Insulin", "type": "number", "step": "1", "min": "0", "max": "900", "placeholder": "110", "default": "110", "description": "Serum insulin concentration (mu U/ml)"},
            {"name": "BMI", "label": "Body Mass Index (BMI)", "type": "number", "step": "0.1", "min": "10.0", "max": "70.0", "placeholder": "32.4", "default": "32.4", "description": "Weight in kg / (height in m)²"},
            {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "step": "0.001", "min": "0.05", "max": "2.5", "placeholder": "0.47", "default": "0.47", "description": "Genetic lineage risk score"},
            {"name": "Age", "label": "Age (years)", "type": "number", "step": "1", "min": "18", "max": "100", "placeholder": "33", "default": "33", "description": "Patient age in completed years"}
        ],
        "sample_inputs": {
            "Pregnancies": 4, "Glucose": 156, "BloodPressure": 80, "SkinThickness": 32,
            "Insulin": 170, "BMI": 35.8, "DiabetesPedigreeFunction": 0.62, "Age": 45
        }
    },
    "naive-bayes": {
        "id": "naive-bayes",
        "name": "Naive Bayes Classifier",
        "category": "Natural Language Processing",
        "lab_experiment": "ML Lab Experiment 08: Naive Bayes Text Classification",
        "model_file": "nb_model.pkl",
        "dataset_file": "college_messages.csv",
        "dataset_name": "College Notification & Communication Dataset (229 Messages)",
        "summary": "Probabilistic text classifier applying Bayes' Theorem with feature independence assumptions to classify campus communications into Important, Promotional, or Spam categories.",
        "theory": "Computes posterior probability P(C|W) ∝ P(C)·Π P(w_i|C) using Multinomial Bag-of-Words token counts and Laplace smoothing to handle unseen vocabulary.",
        "target_type": "categorical",
        "target_label": "Message Category",
        "features": [
            {
                "name": "message",
                "label": "Enter College Message / Notification Text",
                "type": "textarea",
                "rows": 4,
                "placeholder": "e.g. The semester exam timetable is published. Check hall ticket details at the college exam portal.",
                "default": "The semester exam timetable is published. Check hall ticket details at the college exam portal.",
                "description": "Any official circular, promotional offer, or unsolicited message"
            }
        ],
        "sample_inputs": {
            "message": "URGENT: Submit your machine learning lab assignment report before Friday 5 PM to the department coordinator."
        }
    },
    "linear-svm": {
        "id": "linear-svm",
        "name": "Linear Support Vector Machine (SVM)",
        "category": "Classification (Binary)",
        "lab_experiment": "ML Lab Experiment 09: Linear SVM Classifier",
        "model_file": "linear_svm_model.pkl",
        "dataset_file": "data_banknote_authentication.csv",
        "dataset_name": "Banknote Authentication Wavelet Dataset (1,372 Samples)",
        "summary": "Maximizes the geometric margin between genuine and forged banknote classes in continuous 4D wavelet transform feature space.",
        "theory": "Finds the optimal separating hyperplane w·x + b = 0 that maximizes the margin 2 / ||w|| subject to y_i(w·x_i + b) ≥ 1, yielding high generalizability.",
        "target_type": "categorical",
        "target_label": "Banknote Authenticity",
        "features": [
            {"name": "variance", "label": "Wavelet Variance", "type": "number", "step": "0.01", "placeholder": "3.62", "default": "3.62", "description": "Variance of continuous wavelet transform image"},
            {"name": "skewness", "label": "Wavelet Skewness", "type": "number", "step": "0.01", "placeholder": "8.66", "default": "8.66", "description": "Skewness of wavelet transform image"},
            {"name": "curtosis", "label": "Wavelet Curtosis", "type": "number", "step": "0.01", "placeholder": "-2.80", "default": "-2.80", "description": "Curtosis of wavelet transform image"},
            {"name": "entropy", "label": "Image Entropy", "type": "number", "step": "0.01", "placeholder": "-0.44", "default": "-0.44", "description": "Information entropy of genuine/forged specimen"}
        ],
        "sample_inputs": {
            "variance": 2.14, "skewness": 6.82, "curtosis": -1.45, "entropy": -0.85
        }
    },
    "nonlinear-svm": {
        "id": "nonlinear-svm",
        "name": "Non-linear Support Vector Machine",
        "category": "Classification (Multiclass)",
        "lab_experiment": "ML Lab Experiment 10: Non-Linear Support Vector Machine (RBF Kernel)",
        "model_file": "nonlinear_svm_model.pkl",
        "dataset_file": "iris.csv",
        "dataset_name": "Iris Flower Botanical Dataset (150 Samples)",
        "summary": "Applies a non-linear Radial Basis Function (RBF) kernel trick to map non-linearly separable botanical features into infinite-dimensional Hilbert space for maximum margin separation.",
        "theory": "Evaluates kernel function K(x, x') = exp(-γ ||x - x'||²) to form curved, flexible decision boundaries capable of separating non-linearly intertwined classes.",
        "target_type": "categorical",
        "target_label": "Iris Flower Species",
        "features": [
            {"name": "sepal_length", "label": "Sepal Length (cm)", "type": "number", "step": "0.1", "min": "0.1", "max": "15.0", "placeholder": "5.7", "default": "5.7", "description": "Length of sepal in centimeters"},
            {"name": "sepal_width", "label": "Sepal Width (cm)", "type": "number", "step": "0.1", "min": "0.1", "max": "10.0", "placeholder": "2.8", "default": "2.8", "description": "Width of sepal in centimeters"},
            {"name": "petal_length", "label": "Petal Length (cm)", "type": "number", "step": "0.1", "min": "0.1", "max": "15.0", "placeholder": "4.1", "default": "4.1", "description": "Length of petal in centimeters"},
            {"name": "petal_width", "label": "Petal Width (cm)", "type": "number", "step": "0.1", "min": "0.1", "max": "10.0", "placeholder": "1.3", "default": "1.3", "description": "Width of petal in centimeters"}
        ],
        "sample_inputs": {
            "sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.4
        }
    },
    "random-forest": {
        "id": "random-forest",
        "name": "Random Forest Classifier",
        "category": "Ensemble Learning",
        "lab_experiment": "ML Lab Experiment 11: Random Forest Classifier",
        "model_file": "rf_classifier.joblib",
        "dataset_file": "diabetes.csv",
        "dataset_name": "Diabetes Diagnostic Dataset with Imputation (768 Patients)",
        "summary": "Ensemble classifier combining 300 decorrelated decision trees using bootstrap aggregation (bagging) and feature sub-sampling to mitigate overfitting.",
        "theory": "Constructs an ensemble of B=300 trees fitted on bootstrap subsets with random feature subsets at each split; final prediction is aggregated via majority voting P(y) = 1/B Σ T_b(x).",
        "target_type": "categorical",
        "target_label": "Diabetes Diagnosis",
        "features": [
            {"name": "Pregnancies", "label": "Pregnancies Count", "type": "number", "step": "1", "min": "0", "max": "20", "placeholder": "3", "default": "3", "description": "Number of pregnancies"},
            {"name": "Glucose", "label": "Plasma Glucose Concentration", "type": "number", "step": "1", "min": "0", "max": "300", "placeholder": "145", "default": "145", "description": "Oral glucose concentration in mg/dl"},
            {"name": "BloodPressure", "label": "Diastolic Blood Pressure", "type": "number", "step": "1", "min": "0", "max": "150", "placeholder": "76", "default": "76", "description": "Diastolic pressure in mm Hg"},
            {"name": "SkinThickness", "label": "Triceps Skinfold Thickness", "type": "number", "step": "1", "min": "0", "max": "100", "placeholder": "28", "default": "28", "description": "Triceps fold measurement in mm"},
            {"name": "Insulin", "label": "2-Hour Serum Insulin", "type": "number", "step": "1", "min": "0", "max": "900", "placeholder": "125", "default": "125", "description": "Insulin level in mu U/ml"},
            {"name": "BMI", "label": "Body Mass Index (BMI)", "type": "number", "step": "0.1", "min": "10.0", "max": "70.0", "placeholder": "33.2", "default": "33.2", "description": "Weight in kg / (height in m)²"},
            {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "step": "0.001", "min": "0.05", "max": "2.5", "placeholder": "0.52", "default": "0.52", "description": "Hereditary diabetes likelihood factor"},
            {"name": "Age", "label": "Patient Age", "type": "number", "step": "1", "min": "18", "max": "100", "placeholder": "38", "default": "38", "description": "Patient age in years"}
        ],
        "sample_inputs": {
            "Pregnancies": 1, "Glucose": 89, "BloodPressure": 66, "SkinThickness": 23,
            "Insulin": 94, "BMI": 28.1, "DiabetesPedigreeFunction": 0.167, "Age": 21
        }
    }
}


def get_model(algo_id: str) -> Any:
    """Safely loads or retrieves cached model object."""
    if algo_id in _LOADED_MODELS:
        return _LOADED_MODELS[algo_id]

    cfg = ALGORITHM_CONFIGS.get(algo_id)
    if not cfg:
        raise ValueError(f"Unknown algorithm ID: {algo_id}")

    model_path = os.path.join(MODELS_DIR, cfg["model_file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading model {cfg['model_file']}: {e}")

    _LOADED_MODELS[algo_id] = model
    return model


def get_nb_vectorizer():
    """Loads the text CountVectorizer for Naive Bayes."""
    if "nb_vectorizer" in _LOADED_MODELS:
        return _LOADED_MODELS["nb_vectorizer"]

    vec_path = os.path.join(MODELS_DIR, "nb_vectorizer.pkl")
    try:
        vec = joblib.load(vec_path)
    except Exception:
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)

    _LOADED_MODELS["nb_vectorizer"] = vec
    return vec


def execute_prediction(algo_id: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes actual model prediction with proper preprocessing and returns structured result.
    """
    cfg = ALGORITHM_CONFIGS[algo_id]
    model = get_model(algo_id)

    if algo_id == "simple-linear-regression":
        cgpa = float(form_data["cgpa"])
        df_in = pd.DataFrame([[cgpa]], columns=["cgpa"])
        pred_val = float(model.predict(df_in)[0])
        return {
            "result_text": f"{pred_val:.2f} LPA",
            "explanation": f"Based on student CGPA of {cgpa:.2f}, the predicted placement package is {pred_val:.2f} LPA.",
            "badge_class": "badge-accent",
            "numeric_value": pred_val,
            "inputs": {"cgpa": cgpa}
        }

    elif algo_id == "multiple-linear-regression":
        cgpa = float(form_data["cgpa"])
        iq = float(form_data["iq"])
        df_in = pd.DataFrame([[cgpa, iq]], columns=["cgpa", "iq"])
        pred_val = float(model.predict(df_in)[0])
        return {
            "result_text": f"{pred_val:.2f} LPA",
            "explanation": f"For CGPA {cgpa:.2f} and IQ {int(iq)}, estimated campus package is {pred_val:.2f} LPA.",
            "badge_class": "badge-accent",
            "numeric_value": pred_val,
            "inputs": {"cgpa": cgpa, "iq": iq}
        }

    elif algo_id == "binary-logistic-regression":
        cgpa = float(form_data["cgpa"])
        iq = float(form_data["iq"])
        df_in = pd.DataFrame([[cgpa, iq]], columns=["cgpa", "iq"])
        pred = model.predict(df_in)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_in)[0]
            prob = float(probs[1])

        placed = int(pred) == 1
        return {
            "result_text": "Student is Likely to be PLACED" if placed else "Student is NOT Likely to be PLACED",
            "explanation": f"Probability of placement: {prob:.1%}" if prob is not None else "Classified via logistic boundary",
            "badge_class": "badge-success" if placed else "badge-danger",
            "is_positive": placed,
            "confidence": f"{prob:.1%}" if prob is not None else None,
            "inputs": {"cgpa": cgpa, "iq": iq}
        }

    elif algo_id == "multiclass-logistic-regression":
        sl = float(form_data["sepal_length"])
        sw = float(form_data["sepal_width"])
        pl = float(form_data["petal_length"])
        pw = float(form_data["petal_width"])

        cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        df_in = pd.DataFrame([[sl, sw, pl, pw]], columns=cols)
        pred = int(model.predict(df_in)[0])
        species_map = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
        flower = species_map.get(pred, f"Class {pred}")

        probs = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(df_in)[0]
            probs = {species_map[i]: f"{float(prob):.1%}" for i, prob in enumerate(p)}

        return {
            "result_text": flower,
            "explanation": f"Morphological measurements match species characteristics of {flower}.",
            "badge_class": "badge-accent",
            "probabilities": probs,
            "inputs": {"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}
        }

    elif algo_id == "decision-tree":
        feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        values = [float(form_data[f]) for f in feature_order]
        df_in = pd.DataFrame([values], columns=feature_order)

        pred = int(model.predict(df_in)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_in)[0]
            prob = float(probs[pred])

        detected = (pred == 1)
        return {
            "result_text": "Heart Disease Detected" if detected else "No Heart Disease Detected",
            "explanation": f"Tree decision path concluded with {prob * 100:.1f}% confidence." if prob else "Evaluation complete.",
            "badge_class": "badge-danger" if detected else "badge-success",
            "confidence": f"{prob * 100:.1f}%" if prob else None,
            "inputs": dict(zip(feature_order, values))
        }

    elif algo_id == "knn":
        feature_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        values = [float(form_data[f]) for f in feature_order]
        df_in = pd.DataFrame([values], columns=feature_order)

        # Pipeline standardizes and applies KNN
        pred = int(model.predict(df_in)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_in)[0]
            prob = float(probs[pred])

        positive = (pred == 1)
        return {
            "result_text": "Diabetes Detected (Positive)" if positive else "No Diabetes Detected (Negative)",
            "explanation": f"Euclidean distance nearest neighbors classified with {prob:.1%} neighborhood consensus." if prob else "Nearest neighbors evaluated.",
            "badge_class": "badge-danger" if positive else "badge-success",
            "confidence": f"{prob:.1%}" if prob else None,
            "inputs": dict(zip(feature_order, values))
        }

    elif algo_id == "naive-bayes":
        message = form_data.get("message", "").strip()
        if not message:
            raise ValueError("Message text cannot be empty.")

        vectorizer = get_nb_vectorizer()
        vec = vectorizer.transform([message])
        pred = str(model.predict(vec)[0])

        probs = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(vec)[0]
            classes = model.classes_
            probs = {str(c): f"{float(prob):.1%}" for c, prob in zip(classes, p)}

        badge = "badge-success" if pred == "Important" else ("badge-accent" if pred == "Promotional" else "badge-danger")
        return {
            "result_text": f"{pred} Message",
            "explanation": f"Classified as '{pred}' based on Multinomial Bag-of-Words word token likelihoods.",
            "badge_class": badge,
            "probabilities": probs,
            "inputs": {"message": message}
        }

    elif algo_id == "linear-svm":
        feature_order = ["variance", "skewness", "curtosis", "entropy"]
        values = [float(form_data[f]) for f in feature_order]
        # model expects dataframe with column names or array
        df_in = pd.DataFrame([values], columns=feature_order)
        pred = int(model.predict(df_in)[0])

        genuine = (pred == 0)
        return {
            "result_text": "Genuine Banknote" if genuine else "Counterfeit Banknote",
            "explanation": "Separating hyperplane margin confirms banknote is genuine." if genuine else "Feature vector falls within counterfeit boundary.",
            "badge_class": "badge-success" if genuine else "badge-danger",
            "inputs": dict(zip(feature_order, values))
        }

    elif algo_id == "nonlinear-svm":
        values = [
            float(form_data["sepal_length"]),
            float(form_data["sepal_width"]),
            float(form_data["petal_length"]),
            float(form_data["petal_width"])
        ]
        pred = int(model.predict([values])[0])
        flower_names = ["Setosa", "Versicolor", "Virginica"]
        species = flower_names[pred] if pred < len(flower_names) else f"Class {pred}"

        return {
            "result_text": f"Iris {species}",
            "explanation": f"RBF kernel decision function localized sample into Iris {species} region.",
            "badge_class": "badge-accent",
            "inputs": {
                "sepal_length": values[0],
                "sepal_width": values[1],
                "petal_length": values[2],
                "petal_width": values[3]
            }
        }

    elif algo_id == "random-forest":
        feature_order = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        values = {f: float(form_data[f]) for f in feature_order}

        # Model is dict with 'pipeline' and 'features'
        pipeline = model["pipeline"]
        df_input = pd.DataFrame([[values[f] for f in feature_order]], columns=feature_order)

        # Apply same missing-value zero substitution used during training
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_as_missing:
            if col in df_input.columns:
                df_input.loc[df_input[col] == 0, col] = np.nan

        pred = int(pipeline.predict(df_input)[0])
        prob = float(pipeline.predict_proba(df_input)[0, 1])

        positive = (pred == 1)
        return {
            "result_text": "Diabetes-Positive Risk (Class 1)" if positive else "Diabetes-Negative Risk (Class 0)",
            "explanation": f"Ensemble vote across 300 decision trees yielded a {prob:.1%} diabetic risk probability.",
            "badge_class": "badge-danger" if positive else "badge-success",
            "confidence": f"{prob:.1%}",
            "inputs": values
        }

    raise NotImplementedError(f"Predict logic for {algo_id} not implemented.")
