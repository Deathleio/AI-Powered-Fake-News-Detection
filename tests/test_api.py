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

    def test_empty_request_validation(self):
        response = self.client.post("/predict", json={"title": "", "text": ""})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
