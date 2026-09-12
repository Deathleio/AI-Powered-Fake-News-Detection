import unittest
from fastapi.testclient import TestClient
from src.serving.api import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_model_info_endpoint(self):
        response = self.client.get("/api/v1/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["framework"], "PyTorch 2.x")
        self.assertIn("Deep Learning", data["primary_model_type"])
        self.assertIn("Bidirectional LSTM", data["architecture"])

    def test_predict_endpoint_with_dl(self):
        response = self.client.post("/predict", json={
            "title": "Federal Reserve Holds Benchmark Interest Rates Steady Amid Stable Economic Growth",
            "text": "The central bank maintained its benchmark rate in an official policy statement."
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["verdict"], "Real News")
        self.assertFalse(data["is_fake"])
        self.assertLess(data["fake_probability"], 0.50)

    def test_ground_claim_endpoint(self):
        response = self.client.post("/api/v1/ground-claim", json={
            "claim": "Federal Reserve holds benchmark interest rates steady"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("corroboration_score", data)
        self.assertIn("grounding_verdict", data)
        self.assertIsInstance(data["wire_articles"], list)

if __name__ == '__main__':
    unittest.main()
