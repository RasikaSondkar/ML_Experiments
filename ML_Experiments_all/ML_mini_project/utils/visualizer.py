import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

from .evaluator import evaluate_model, get_dataset
from .model_loader import get_model


# Academic muted blue color scheme
PRIMARY_COLOR = "#1e40af"     # Muted Royal Blue
ACCENT_COLOR = "#3b82f6"      # Soft Sky Blue
ACCENT_LIGHT = "#eff6ff"      # Off-white / light blue tint
TEXT_COLOR = "#1f2937"        # Dark slate
GRID_COLOR = "#e5e7eb"        # Subtle gray grid
SUCCESS_COLOR = "#10b981"     # Emerald Green
DANGER_COLOR = "#ef4444"      # Crimson Red


def _fig_to_base64(fig) -> str:
    """Converts a matplotlib figure to base64 data URI string."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def generate_visualization(algo_id: str, user_inputs: Optional[Dict[str, Any]] = None, user_prediction: Optional[Any] = None) -> str:
    """
    Generates a relevant, clean visualization for the given algorithm.
    Integrates the user's specific prediction point if available.
    """
    eval_res = evaluate_model(algo_id)

    # Global font and style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.edgecolor': '#cbd5e1',
        'axes.linewidth': 0.8
    })

    if algo_id == "simple-linear-regression":
        df = get_dataset(algo_id)
        model = get_model(algo_id)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(df['cgpa'], df['package'], color='#93c5fd', edgecolors=PRIMARY_COLOR, alpha=0.7, s=40, label='Student Records')

        # Regression Line
        cgpa_range = np.linspace(df['cgpa'].min(), df['cgpa'].max(), 100).reshape(-1, 1)
        pkg_pred = model.predict(cgpa_range)
        ax.plot(cgpa_range, pkg_pred, color=PRIMARY_COLOR, linewidth=2.2, label='Fitted Regression Line')

        # Highlight user input point if available
        if user_inputs and 'cgpa' in user_inputs and user_prediction is not None:
            user_cgpa = float(user_inputs['cgpa'])
            user_pkg = float(user_prediction['numeric_value'])
            ax.scatter([user_cgpa], [user_pkg], color=DANGER_COLOR, s=120, zorder=5, edgecolors='black', linewidth=1.5, label=f'Your Input ({user_cgpa:.2f}, {user_pkg:.2f})')
            ax.axvline(x=user_cgpa, color=DANGER_COLOR, linestyle=':', alpha=0.6)
            ax.axhline(y=user_pkg, color=DANGER_COLOR, linestyle=':', alpha=0.6)

        ax.set_title("Simple Linear Regression: CGPA vs. Placement Package", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Student CGPA (0.0 to 10.0)", fontsize=10)
        ax.set_ylabel("Package (in LPA)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9)
        return _fig_to_base64(fig)

    elif algo_id == "multiple-linear-regression":
        y_true = eval_res["y_true"]
        y_pred = eval_res["y_pred"]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(y_true, y_pred, color='#60a5fa', edgecolors=PRIMARY_COLOR, alpha=0.7, s=45, label='Student Observations')

        # Perfect prediction diagonal
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color=DANGER_COLOR, linestyle='--', linewidth=1.8, label='Ideal Perfect Fit (y = x)')

        if user_prediction and 'numeric_value' in user_prediction:
            user_pkg = float(user_prediction['numeric_value'])
            ax.axhline(y=user_pkg, color=SUCCESS_COLOR, linestyle=':', linewidth=1.5, label=f'Current Prediction: {user_pkg:.2f} LPA')

        ax.set_title("Multiple Linear Regression: Actual vs. Predicted Package", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Actual Package (LPA)", fontsize=10)
        ax.set_ylabel("Model Predicted Package (LPA)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9)
        return _fig_to_base64(fig)

    elif algo_id == "binary-logistic-regression":
        df = get_dataset(algo_id)
        fig, ax = plt.subplots(figsize=(7, 4.2))

        # Plot training data points
        not_placed = df[df['placement'] == 0]
        placed = df[df['placement'] == 1]

        ax.scatter(not_placed['cgpa'], not_placed['iq'], color='#f87171', edgecolors='#991b1b', s=45, alpha=0.8, label='Not Placed (0)')
        ax.scatter(placed['cgpa'], placed['iq'], color='#34d399', edgecolors='#065f46', s=45, alpha=0.8, label='Placed (1)')

        # User point
        if user_inputs and 'cgpa' in user_inputs and 'iq' in user_inputs:
            u_cgpa = float(user_inputs['cgpa'])
            u_iq = float(user_inputs['iq'])
            ax.scatter([u_cgpa], [u_iq], color=PRIMARY_COLOR, s=140, edgecolors='black', linewidth=1.8, zorder=6, label=f'Your Input ({u_cgpa}, {u_iq})')

        ax.set_title("Binary Logistic Regression: Placement Decision Space (CGPA vs. IQ)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("CGPA", fontsize=10)
        ax.set_ylabel("IQ Score", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9)
        return _fig_to_base64(fig)

    elif algo_id in ["multiclass-logistic-regression", "nonlinear-svm"]:
        cm = np.array(eval_res["confusion_matrix"])
        classes = eval_res["classes"]

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        algo_title = "Multiclass Logistic Regression" if algo_id == "multiclass-logistic-regression" else "Non-linear SVM (RBF)"
        ax.set_title(f"{algo_title}: Confusion Matrix", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("Actual Ground Truth", fontsize=10)
        return _fig_to_base64(fig)

    elif algo_id == "decision-tree":
        # Feature importances plot
        importances = eval_res.get("feature_importances", {})
        if importances:
            sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
            feat_names = [k for k, v in sorted_items][::-1]
            feat_scores = [v for k, v in sorted_items][::-1]

            fig, ax = plt.subplots(figsize=(7, 4.2))
            bars = ax.barh(feat_names, feat_scores, color='#3b82f6', edgecolor=PRIMARY_COLOR, height=0.6)
            ax.set_title("Decision Tree: Top Clinical Feature Importances (Gini)", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Normalized Gini Importance", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='x')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                        va='center', ha='left', fontsize=8.5, color=TEXT_COLOR)
            return _fig_to_base64(fig)
        else:
            cm = np.array(eval_res["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title("Decision Tree: Confusion Matrix", fontsize=12, fontweight='bold')
            return _fig_to_base64(fig)

    elif algo_id == "random-forest":
        importances = eval_res.get("feature_importances", {})
        if importances:
            sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            feat_names = [k for k, v in sorted_items][::-1]
            feat_scores = [v for k, v in sorted_items][::-1]

            fig, ax = plt.subplots(figsize=(7, 4.2))
            bars = ax.barh(feat_names, feat_scores, color='#2563eb', edgecolor='#1e3a8a', height=0.6)
            ax.set_title("Random Forest: Ensemble Feature Importance Ranking", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("Mean Impurity Reduction", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='x')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                        va='center', ha='left', fontsize=8.5, color=TEXT_COLOR)
            return _fig_to_base64(fig)

    elif algo_id == "linear-svm":
        df = get_dataset(algo_id)
        fig, ax = plt.subplots(figsize=(7, 4.2))

        # Plot sample of genuine vs counterfeit
        sample_df = df.sample(min(300, len(df)), random_state=42)
        genuine = sample_df[sample_df['class'] == 0]
        counterfeit = sample_df[sample_df['class'] == 1]

        ax.scatter(genuine['variance'], genuine['skewness'], color='#60a5fa', edgecolors='#1e40af', alpha=0.7, s=40, label='Genuine (0)')
        ax.scatter(counterfeit['variance'], counterfeit['skewness'], color='#f87171', edgecolors='#991b1b', alpha=0.7, s=40, label='Counterfeit (1)')

        if user_inputs and 'variance' in user_inputs and 'skewness' in user_inputs:
            u_var = float(user_inputs['variance'])
            u_skew = float(user_inputs['skewness'])
            ax.scatter([u_var], [u_skew], color=SUCCESS_COLOR, s=150, edgecolors='black', linewidth=1.8, zorder=6, label=f'Your Banknote ({u_var}, {u_skew})')

        ax.set_title("Linear SVM: Banknote Wavelet Variance vs. Skewness Projection", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Wavelet Variance", fontsize=10)
        ax.set_ylabel("Wavelet Skewness", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9)
        return _fig_to_base64(fig)

    # Default fallback: Confusion Matrix for any classification task
    cm = np.array(eval_res.get("confusion_matrix", [[0, 0], [0, 0]]))
    classes = eval_res.get("classes", ["Class 0", "Class 1"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title(f"{eval_res.get('algo_id', algo_id).title()}: Evaluation Confusion Matrix", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Ground Truth", fontsize=10)
    return _fig_to_base64(fig)
