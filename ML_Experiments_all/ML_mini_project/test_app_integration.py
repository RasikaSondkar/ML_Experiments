import unittest
import os
import sys

# Ensure project root is in path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from app import app
from utils.model_loader import ALGORITHM_CONFIGS, get_model, execute_prediction
from utils.evaluator import evaluate_model, get_dataset_preview
from utils.visualizer import generate_visualization


class TestAlgorithmDashboard(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_routes_exist(self):
        """Test home and about page routes."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Integrated Machine Learning Algorithm Dashboard", res.data)

        res_about = self.client.get('/about')
        self.assertEqual(res_about.status_code, 200)
        self.assertIn(b"About the Project", res_about.data)

    def test_all_10_algorithms_load_and_evaluate(self):
        """Test that all 10 algorithms load models, evaluate datasets, and render views."""
        for algo_id, cfg in ALGORITHM_CONFIGS.items():
            # 1. Model can be loaded
            model = get_model(algo_id)
            self.assertIsNotNone(model, f"Model for {algo_id} should not be None")

            # 2. Dataset can be previewed
            preview = get_dataset_preview(algo_id)
            self.assertGreater(preview["total_rows"], 0, f"Dataset for {algo_id} should have rows")

            # 3. Model can be evaluated on dataset
            eval_res = evaluate_model(algo_id)
            self.assertIn("metrics", eval_res, f"Evaluation for {algo_id} must have metrics")
            self.assertGreater(len(eval_res["metrics"]), 0)

            # 4. View page returns 200
            res = self.client.get(f'/algorithm/{algo_id}')
            self.assertEqual(res.status_code, 200, f"Algorithm page {algo_id} failed with {res.status_code}")
            self.assertIn(cfg["name"].encode(), res.data)

            # 5. Visualization generates clean Base64 image
            chart = generate_visualization(algo_id)
            self.assertTrue(chart.startswith("data:image/png;base64,"), f"Chart for {algo_id} should be base64 PNG")

    def test_all_10_algorithm_predictions(self):
        """Test POST /predict with sample data for each of the 10 algorithms."""
        for algo_id, cfg in ALGORITHM_CONFIGS.items():
            sample_data = cfg.get("sample_inputs", {})
            self.assertTrue(bool(sample_data), f"Algorithm {algo_id} must have sample inputs")

            # Submit prediction via Flask client
            res = self.client.post(f'/predict/{algo_id}', data=sample_data)
            self.assertEqual(res.status_code, 200, f"Prediction for {algo_id} failed with {res.status_code}")
            self.assertIn(b"id=\"prediction-result-container\"", res.data)

    def test_input_validation_missing_fields(self):
        """Test that missing inputs trigger 400 Bad Request with friendly error."""
        res = self.client.post('/predict/simple-linear-regression', data={})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Missing required input values", res.data)

    def test_input_validation_invalid_numeric(self):
        """Test that non-numeric inputs trigger 400 Bad Request."""
        res = self.client.post('/predict/simple-linear-regression', data={'cgpa': 'invalid_string'})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Invalid numeric input", res.data)


if __name__ == '__main__':
    unittest.main()
