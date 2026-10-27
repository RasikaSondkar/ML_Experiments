import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, abort

from utils.model_loader import ALGORITHM_CONFIGS, execute_prediction
from utils.evaluator import evaluate_model, get_dataset_preview
from utils.visualizer import generate_visualization

app = Flask(__name__)
app.secret_key = "academic-ml-dashboard-secret-key"


@app.context_processor
def inject_global_vars():
    """Provides algorithm list to all templates for dynamic sidebar navigation."""
    algorithms = list(ALGORITHM_CONFIGS.values())
    return {
        "sidebar_algorithms": algorithms
    }


@app.route("/")
def index():
    """Home dashboard listing all 10 integrated machine learning algorithms."""
    algorithms = list(ALGORITHM_CONFIGS.values())
    return render_template(
        "index.html",
        active_page="home",
        algorithms=algorithms
    )


@app.route("/algorithm/<algo_id>", methods=["GET"])
def algorithm_view(algo_id):
    """Renders the detailed interactive workspace for an individual algorithm."""
    if algo_id not in ALGORITHM_CONFIGS:
        abort(404)

    algo = ALGORITHM_CONFIGS[algo_id]

    try:
        eval_data = evaluate_model(algo_id)
        dataset_preview = get_dataset_preview(algo_id, limit=5)
        chart_base64 = generate_visualization(algo_id)
    except Exception as e:
        app.logger.error(f"Error preparing algorithm view for {algo_id}: {e}")
        return render_template(
            "algorithm.html",
            active_page=algo_id,
            algo=algo,
            error=f"Error evaluating dataset: {str(e)}",
            eval_data={"metrics": [], "total_samples": 0},
            dataset_preview={"columns": [], "rows": [], "total_rows": 0, "total_cols": 0},
            chart_base64="",
            sample_json=json.dumps(algo.get("sample_inputs", {})),
            submitted_inputs=None,
            prediction=None
        )

    sample_json = json.dumps(algo.get("sample_inputs", {}))

    return render_template(
        "algorithm.html",
        active_page=algo_id,
        algo=algo,
        eval_data=eval_data,
        dataset_preview=dataset_preview,
        chart_base64=chart_base64,
        sample_json=sample_json,
        submitted_inputs=None,
        prediction=None,
        error=None
    )


@app.route("/predict/<algo_id>", methods=["POST"])
def predict(algo_id):
    """Validates submitted inputs, executes model inference, and renders prediction."""
    if algo_id not in ALGORITHM_CONFIGS:
        abort(404)

    algo = ALGORITHM_CONFIGS[algo_id]
    form_data = request.form.to_dict()

    try:
        eval_data = evaluate_model(algo_id)
        dataset_preview = get_dataset_preview(algo_id, limit=5)
    except Exception as e:
        eval_data = {"metrics": [], "total_samples": 0}
        dataset_preview = {"columns": [], "rows": [], "total_rows": 0, "total_cols": 0}

    sample_json = json.dumps(algo.get("sample_inputs", {}))

    # Basic input validation
    missing_fields = []
    for feat in algo["features"]:
        val = form_data.get(feat["name"], "").strip()
        if val == "":
            missing_fields.append(feat["label"])

    if missing_fields:
        error_msg = f"Missing required input values for: {', '.join(missing_fields)}"
        chart_base64 = generate_visualization(algo_id)
        return render_template(
            "algorithm.html",
            active_page=algo_id,
            algo=algo,
            eval_data=eval_data,
            dataset_preview=dataset_preview,
            chart_base64=chart_base64,
            sample_json=sample_json,
            submitted_inputs=form_data,
            prediction=None,
            error=error_msg
        ), 400

    try:
        # Execute genuine prediction
        prediction = execute_prediction(algo_id, form_data)
        # Generate chart with user prediction point embedded
        chart_base64 = generate_visualization(algo_id, user_inputs=form_data, user_prediction=prediction)

        return render_template(
            "algorithm.html",
            active_page=algo_id,
            algo=algo,
            eval_data=eval_data,
            dataset_preview=dataset_preview,
            chart_base64=chart_base64,
            sample_json=sample_json,
            submitted_inputs=form_data,
            prediction=prediction,
            error=None
        )

    except ValueError as e:
        error_msg = f"Invalid numeric input: {str(e)}. Please check your values."
        chart_base64 = generate_visualization(algo_id)
        return render_template(
            "algorithm.html",
            active_page=algo_id,
            algo=algo,
            eval_data=eval_data,
            dataset_preview=dataset_preview,
            chart_base64=chart_base64,
            sample_json=sample_json,
            submitted_inputs=form_data,
            prediction=None,
            error=error_msg
        ), 400
    except Exception as e:
        app.logger.error(f"Inference error on {algo_id}: {e}")
        error_msg = f"Prediction failed: {str(e)}"
        chart_base64 = generate_visualization(algo_id)
        return render_template(
            "algorithm.html",
            active_page=algo_id,
            algo=algo,
            eval_data=eval_data,
            dataset_preview=dataset_preview,
            chart_base64=chart_base64,
            sample_json=sample_json,
            submitted_inputs=form_data,
            prediction=None,
            error=error_msg
        ), 500


@app.route("/about")
def about():
    """Renders the About Project and Viva guide page."""
    return render_template(
        "about.html",
        active_page="about"
    )


@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "base.html",
        active_page=""
    ), 404


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  INTEGRATED MACHINE LEARNING ALGORITHM DASHBOARD")
    print("  Local College Mini Project Server")
    print("=" * 60)
    print("  Running at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to terminate\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
