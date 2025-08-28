"""
Pydantic schemas for the scoring API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime


class ScoreRequest(BaseModel):
    """Request schema for anomaly scoring."""
    
    device_id: str = Field(..., description="Device identifier")
    window_end: str = Field(..., description="Window end timestamp (ISO format)")
    features: Dict[str, float] = Field(..., description="Feature values")
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature values."""
        required_features = ['temperature', 'humidity']
        for feat in required_features:
            if feat not in v:
                raise ValueError(f"Missing required feature: {feat}")
        return v
    
    @validator('window_end')
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Invalid timestamp format. Use ISO format.")
        return v


class ScoreResponse(BaseModel):
    """Response schema for anomaly scoring."""
    
    device_id: str = Field(..., description="Device identifier")
    event_ts: str = Field(..., description="Event timestamp")
    score: float = Field(..., description="Anomaly score")
    threshold: float = Field(..., description="Anomaly threshold")
    is_anomaly: bool = Field(..., description="Whether input is anomalous")
    contributions: Dict[str, float] = Field(..., description="Per-feature contributions")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class BatchScoreRequest(BaseModel):
    """Request schema for batch scoring."""
    
    requests: List[ScoreRequest] = Field(..., description="List of score requests")
    
    @validator('requests')
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100")
        return v


class BatchScoreResponse(BaseModel):
    """Response schema for batch scoring."""
    
    responses: List[ScoreResponse] = Field(..., description="List of score responses")
    batch_processing_time_ms: float = Field(..., description="Total batch processing time")


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")


class MetricsResponse(BaseModel):
    """Metrics response schema."""
    
    total_requests: int = Field(..., description="Total requests processed")
    total_anomalies: int = Field(..., description="Total anomalies detected")
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    p50_processing_time_ms: float = Field(..., description="P50 processing time")
    p95_processing_time_ms: float = Field(..., description="P95 processing time")
    p99_processing_time_ms: float = Field(..., description="P99 processing time")
    anomaly_rate: float = Field(..., description="Overall anomaly rate")
    requests_per_second: float = Field(..., description="Current requests per second")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


class WindowedFeaturesRequest(BaseModel):
    """Request schema for windowed features (real streaming scenario)."""
    
    device_id: str = Field(..., description="Device identifier")
    window_end: str = Field(..., description="Window end timestamp")
    avg_temp: float = Field(..., description="Average temperature")
    min_temp: float = Field(..., description="Minimum temperature")
    max_temp: float = Field(..., description="Maximum temperature")
    std_temp: float = Field(..., description="Standard deviation of temperature")
    current_temp: float = Field(..., description="Current temperature value")
    avg_humidity: float = Field(..., description="Average humidity")
    min_humidity: float = Field(..., description="Minimum humidity")
    max_humidity: float = Field(..., description="Maximum humidity")
    std_humidity: float = Field(..., description="Standard deviation of humidity")
    current_humidity: float = Field(..., description="Current humidity value")
    
    def to_features_dict(self) -> Dict[str, float]:
        """Convert to features dictionary for scoring."""
        return {
            'avg_temp': self.avg_temp,
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'std_temp': self.std_temp,
            'current_temp': self.current_temp,
            'avg_humidity': self.avg_humidity,
            'min_humidity': self.min_humidity,
            'max_humidity': self.max_humidity,
            'std_humidity': self.std_humidity,
            'current_humidity': self.current_humidity,
            # For compatibility with basic scorer
            'temperature': self.current_temp,
            'humidity': self.current_humidity
        }
