import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, root_mean_squared_error,
    confusion_matrix
)
from .model_loader import ALGORITHM_CONFIGS, get_model, get_nb_vectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

_EVALUATION_CACHE: Dict[str, Any] = {}


def get_dataset(algo_id: str) -> pd.DataFrame:
    """Reads the dataset corresponding to the given algorithm."""
    cfg = ALGORITHM_CONFIGS[algo_id]
    csv_path = os.path.join(DATA_DIR, cfg["dataset_file"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def evaluate_model(algo_id: str) -> Dict[str, Any]:
    """
    Evaluates the loaded model against the genuine dataset.
    Caches the calculated metrics for fast response times.
    """
    if algo_id in _EVALUATION_CACHE:
        return _EVALUATION_CACHE[algo_id]

    cfg = ALGORITHM_CONFIGS[algo_id]
    model = get_model(algo_id)
    df = get_dataset(algo_id)

    res = {
        "algo_id": algo_id,
        "dataset_name": cfg["dataset_name"],
        "dataset_file": cfg["dataset_file"],
        "total_samples": len(df),
        "target_type": cfg["target_type"]
    }

    if algo_id == "simple-linear-regression":
        X = df[['cgpa']]
        y = df['package']
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)

        res["metrics"] = [
            {"name": "R² Determination", "value": f"{r2:.4f}", "desc": "Proportion of package variance explained by CGPA"},
            {"name": "Mean Absolute Error (MAE)", "value": f"{mae:.4f} LPA", "desc": "Average magnitude of salary prediction error"},
            {"name": "Root Mean Squared Error (RMSE)", "value": f"{rmse:.4f} LPA", "desc": "Standard deviation of regression residuals"},
            {"name": "Total Samples", "value": str(len(df)), "desc": "Evaluated college student placement records"}
        ]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "multiple-linear-regression":
        X = df[['cgpa', 'iq']]
        y = df['package']
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)

        res["metrics"] = [
            {"name": "R² Determination", "value": f"{r2:.4f}", "desc": "Combined variance explained by CGPA & IQ"},
            {"name": "Mean Absolute Error (MAE)", "value": f"{mae:.4f} LPA", "desc": "Average magnitude of salary prediction error"},
            {"name": "Root Mean Squared Error (RMSE)", "value": f"{rmse:.4f} LPA", "desc": "Standard deviation of multivariate residuals"},
            {"name": "Total Samples", "value": str(len(df)), "desc": "Evaluated student records"}
        ]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "binary-logistic-regression":
        X = df[['cgpa', 'iq']]
        y = df['placement']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        res["metrics"] = [
            {"name": "Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Overall correct placement classifications"},
            {"name": "Precision", "value": f"{prec * 100:.2f}%", "desc": "Positive predictive value for placements"},
            {"name": "Recall (Sensitivity)", "value": f"{rec * 100:.2f}%", "desc": "True placed students correctly identified"},
            {"name": "F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic mean of precision and recall"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["Not Placed (0)", "Placed (1)"]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "multiclass-logistic-regression":
        feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        X = df[feature_cols]
        y = df['target']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted')
        rec = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)

        res["metrics"] = [
            {"name": "Overall Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Total botanical classifications matching actual species"},
            {"name": "Weighted Precision", "value": f"{prec * 100:.2f}%", "desc": "Weighted average precision across all 3 classes"},
            {"name": "Weighted Recall", "value": f"{rec * 100:.2f}%", "desc": "Weighted average true positive rate"},
            {"name": "Weighted F1-Score", "value": f"{f1:.4f}", "desc": "Weighted harmonic mean"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["Setosa", "Versicolor", "Virginica"]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "decision-tree":
        features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        target_col = [c for c in df.columns if c not in features][0]
        X = df[features]
        y = df[target_col]
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        res["metrics"] = [
            {"name": "Diagnostic Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Proportion of correct heart disease predictions"},
            {"name": "Precision", "value": f"{prec * 100:.2f}%", "desc": "Correctness when heart disease is predicted"},
            {"name": "Recall (Sensitivity)", "value": f"{rec * 100:.2f}%", "desc": "Actual heart disease patients identified"},
            {"name": "F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic balance of precision and recall"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["No Disease (0)", "Disease Present (1)"]
        res["feature_importances"] = dict(zip(features, model.feature_importances_))
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "knn":
        features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        X = df[features]
        y = df['Outcome']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        res["metrics"] = [
            {"name": "Classification Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Accuracy across 768 patient records"},
            {"name": "Precision", "value": f"{prec * 100:.2f}%", "desc": "Precision for positive diabetes class"},
            {"name": "Recall", "value": f"{rec * 100:.2f}%", "desc": "Sensitivity for positive diabetes cases"},
            {"name": "F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic mean of diagnostic precision/recall"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["Negative (0)", "Positive (1)"]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "naive-bayes":
        vectorizer = get_nb_vectorizer()
        X_vec = vectorizer.transform(df["message"])
        y = df["label"]
        y_pred = model.predict(X_vec)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        labels = sorted(list(set(y)))
        cm = confusion_matrix(y, y_pred, labels=labels)

        res["metrics"] = [
            {"name": "Text Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Accuracy across college notification corpus"},
            {"name": "Weighted Precision", "value": f"{prec * 100:.2f}%", "desc": "Precision weighted across message types"},
            {"name": "Weighted Recall", "value": f"{rec * 100:.2f}%", "desc": "Recall weighted across message types"},
            {"name": "Weighted F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic mean for text classification"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = labels
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "linear-svm":
        features = ["variance", "skewness", "curtosis", "entropy"]
        X = df[features]
        y = df["class"]
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        res["metrics"] = [
            {"name": "Classification Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Authenticity detection rate on 1,372 banknotes"},
            {"name": "Precision", "value": f"{prec * 100:.2f}%", "desc": "Precision when flagging counterfeit notes"},
            {"name": "Recall", "value": f"{rec * 100:.2f}%", "desc": "Counterfeit detection sensitivity"},
            {"name": "F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic mean of authentication metric"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["Genuine (0)", "Counterfeit (1)"]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "nonlinear-svm":
        feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        X = df[feature_cols]
        y = df['target']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted')
        rec = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)

        res["metrics"] = [
            {"name": "Non-Linear Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Overall accuracy with RBF kernel on Iris dataset"},
            {"name": "Weighted Precision", "value": f"{prec * 100:.2f}%", "desc": "Precision weighted across 3 species"},
            {"name": "Weighted Recall", "value": f"{rec * 100:.2f}%", "desc": "Recall weighted across 3 species"},
            {"name": "Weighted F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic mean of RBF classification"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["Setosa", "Versicolor", "Virginica"]
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    elif algo_id == "random-forest":
        pipeline = model["pipeline"]
        features = model["features"]
        df_clean = df.drop_duplicates().copy()
        for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            df_clean[col] = df_clean[col].replace(0, np.nan)
        X = df_clean[features]
        y = df_clean["Outcome"]
        y_pred = pipeline.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        # Get feature importances from forest step
        rf_step = pipeline.named_steps.get("model")
        feature_importances = {}
        if hasattr(rf_step, "feature_importances_"):
            feature_importances = dict(zip(features, rf_step.feature_importances_))

        res["metrics"] = [
            {"name": "Ensemble Accuracy", "value": f"{acc * 100:.2f}%", "desc": "Accuracy across 300 bagged decision trees"},
            {"name": "Precision", "value": f"{prec * 100:.2f}%", "desc": "Precision for diabetes positive diagnosis"},
            {"name": "Recall", "value": f"{rec * 100:.2f}%", "desc": "Sensitivity for identifying at-risk patients"},
            {"name": "F1-Score", "value": f"{f1:.4f}", "desc": "Harmonic balance of ensemble predictions"}
        ]
        res["confusion_matrix"] = cm.tolist()
        res["classes"] = ["Negative (0)", "Positive (1)"]
        res["feature_importances"] = feature_importances
        res["y_true"] = y.values
        res["y_pred"] = y_pred

    _EVALUATION_CACHE[algo_id] = res
    return res


def get_dataset_preview(algo_id: str, limit: int = 5) -> Dict[str, Any]:
    """Returns headers and top rows for previewing the dataset in the UI."""
    df = get_dataset(algo_id)
    return {
        "columns": list(df.columns),
        "rows": df.head(limit).values.tolist(),
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }
