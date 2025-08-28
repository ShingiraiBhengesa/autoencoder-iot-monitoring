"""
Contract tests for the serving API.
"""

import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime
from src.serving.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIContract:
    """Test API contract and endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert data["service"] == "IoT Anomaly Detection"
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "memory_usage_mb" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        # Should return either metrics or 503 if collector not available
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "total_requests" in data
            assert "anomaly_rate" in data
            assert "avg_processing_time_ms" in data
    
    def test_score_endpoint_valid_request(self, client):
        """Test score endpoint with valid request."""
        request_data = {
            "device_id": "test-device-1",
            "window_end": datetime.now().isoformat(),
            "features": {
                "temperature": 25.5,
                "humidity": 45.2
            }
        }
        
        response = client.post("/score", json=request_data)
        
        # Should return either score or 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "device_id" in data
            assert "score" in data
            assert "is_anomaly" in data
            assert "contributions" in data
            assert "model_version" in data
            assert data["device_id"] == "test-device-1"
            assert isinstance(data["score"], float)
            assert isinstance(data["is_anomaly"], bool)
    
    def test_score_endpoint_missing_features(self, client):
        """Test score endpoint with missing features."""
        request_data = {
            "device_id": "test-device-1",
            "window_end": datetime.now().isoformat(),
            "features": {
                "temperature": 25.5
                # Missing humidity
            }
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_score_endpoint_invalid_timestamp(self, client):
        """Test score endpoint with invalid timestamp."""
        request_data = {
            "device_id": "test-device-1",
            "window_end": "not-a-timestamp",
            "features": {
                "temperature": 25.5,
                "humidity": 45.2
            }
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_score_endpoint(self, client):
        """Test batch scoring endpoint."""
        request_data = {
            "requests": [
                {
                    "device_id": "test-device-1",
                    "window_end": datetime.now().isoformat(),
                    "features": {"temperature": 25.5, "humidity": 45.2}
                },
                {
                    "device_id": "test-device-2",
                    "window_end": datetime.now().isoformat(),
                    "features": {"temperature": 22.1, "humidity": 50.8}
                }
            ]
        }
        
        response = client.post("/score/batch", json=request_data)
        
        # Should return either scores or 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "responses" in data
            assert "batch_processing_time_ms" in data
            assert len(data["responses"]) == 2
    
    def test_windowed_features_endpoint(self, client):
        """Test windowed features endpoint."""
        request_data = {
            "device_id": "test-device-1",
            "window_end": datetime.now().isoformat(),
            "avg_temp": 25.0,
            "min_temp": 20.0,
            "max_temp": 30.0,
            "std_temp": 2.5,
            "current_temp": 25.5,
            "avg_humidity": 45.0,
            "min_humidity": 40.0,
            "max_humidity": 50.0,
            "std_humidity": 3.0,
            "current_humidity": 45.2
        }
        
        response = client.post("/score/windowed", json=request_data)
        
        # Should return either score or 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "device_id" in data
            assert "score" in data
            assert "is_anomaly" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        # Should return either model info or 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_version" in data
            assert "threshold" in data
            assert "feature_names" in data
            assert "window_size" in data
    
    def test_batch_size_limit(self, client):
        """Test batch size validation."""
        # Create request with too many items
        large_batch = {
            "requests": [
                {
                    "device_id": f"test-device-{i}",
                    "window_end": datetime.now().isoformat(),
                    "features": {"temperature": 25.0, "humidity": 45.0}
                }
                for i in range(101)  # Over the limit of 100
            ]
        }
        
        response = client.post("/score/batch", json=large_batch)
        assert response.status_code == 422  # Validation error
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/score")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_request_logging(self, client):
        """Test that requests are logged (implicit test)."""
        # Make a request and verify it doesn't fail
        response = client.get("/health")
        assert response.status_code == 200
        
        # In a real scenario, you'd check logs or metrics
        # For now, just ensure the middleware doesn't break anything


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling."""
        response = client.post(
            "/score",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_field(self, client):
        """Test missing required field."""
        request_data = {
            # Missing device_id
            "window_end": datetime.now().isoformat(),
            "features": {"temperature": 25.5, "humidity": 45.2}
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_feature_types(self, client):
        """Test invalid feature types."""
        request_data = {
            "device_id": "test-device",
            "window_end": datetime.now().isoformat(),
            "features": {
                "temperature": "not-a-number",  # Should be float
                "humidity": 45.2
            }
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
