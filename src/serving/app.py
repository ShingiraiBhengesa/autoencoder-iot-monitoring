"""
FastAPI scoring service for real-time anomaly detection.
"""

import time
import asyncio
import psutil
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import os
import json

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .schema import (
    ScoreRequest, ScoreResponse, BatchScoreRequest, BatchScoreResponse,
    HealthResponse, MetricsResponse, ErrorResponse, WindowedFeaturesRequest
)
from .onnx_infer import ONNXInferenceEngine
from .metrics import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
inference_engine: Optional[ONNXInferenceEngine] = None
metrics_collector: Optional[MetricsCollector] = None
app_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global inference_engine, metrics_collector
    
    # Startup
    logger.info("Starting anomaly detection service...")
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # Load model artifacts
    model_dir = os.getenv('MODEL_DIR', 'models')
    
    try:
        inference_engine = ONNXInferenceEngine(
            onnx_model_path=os.path.join(model_dir, 'model.onnx'),
            scaler_path=os.path.join(model_dir, 'scaler.pkl'),
            threshold_path=os.path.join(model_dir, 'threshold.json'),
            feature_info_path=os.path.join(model_dir, 'feature_info.json')
        )
        logger.info("ONNX inference engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference engine: {e}")
        # For development, create a mock engine
        inference_engine = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down anomaly detection service...")


# Create FastAPI app
app = FastAPI(
    title="IoT Anomaly Detection Service",
    description="Real-time anomaly detection for IoT sensor data using autoencoders",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.2f}ms"
    )
    
    # Track metrics
    if metrics_collector:
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=process_time
        )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An internal error occurred",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global inference_engine, app_start_time
    
    # Memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Uptime
    uptime_seconds = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        model_version=inference_engine.model_version if inference_engine else "unknown",
        uptime_seconds=uptime_seconds,
        memory_usage_mb=memory_mb
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service metrics."""
    global metrics_collector
    
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not available")
    
    metrics = metrics_collector.get_metrics()
    
    return MetricsResponse(**metrics)


@app.post("/score", response_model=ScoreResponse)
async def score_single(request: ScoreRequest):
    """Score a single feature vector for anomalies."""
    global inference_engine, metrics_collector
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Perform inference
        score, is_anomaly, contributions = inference_engine.score(request.features)
        
        # Track metrics
        processing_time_ms = (time.time() - start_time) * 1000
        
        if metrics_collector:
            metrics_collector.record_scoring(
                is_anomaly=is_anomaly,
                score=score,
                processing_time_ms=processing_time_ms
            )
        
        return ScoreResponse(
            device_id=request.device_id,
            event_ts=request.window_end,
            score=score,
            threshold=inference_engine.threshold,
            is_anomaly=is_anomaly,
            contributions=contributions,
            model_version=inference_engine.model_version,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest):
    """Score a batch of feature vectors."""
    global inference_engine, metrics_collector
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    batch_start_time = time.time()
    responses = []
    
    try:
        for single_request in request.requests:
            start_time = time.time()
            
            # Perform inference
            score, is_anomaly, contributions = inference_engine.score(single_request.features)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track metrics
            if metrics_collector:
                metrics_collector.record_scoring(
                    is_anomaly=is_anomaly,
                    score=score,
                    processing_time_ms=processing_time_ms
                )
            
            responses.append(ScoreResponse(
                device_id=single_request.device_id,
                event_ts=single_request.window_end,
                score=score,
                threshold=inference_engine.threshold,
                is_anomaly=is_anomaly,
                contributions=contributions,
                model_version=inference_engine.model_version,
                processing_time_ms=processing_time_ms
            ))
        
        batch_processing_time_ms = (time.time() - batch_start_time) * 1000
        
        return BatchScoreResponse(
            responses=responses,
            batch_processing_time_ms=batch_processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")


@app.post("/score/windowed", response_model=ScoreResponse)
async def score_windowed_features(request: WindowedFeaturesRequest):
    """Score windowed features (from Stream Analytics)."""
    global inference_engine, metrics_collector
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert windowed request to features dict
        features = request.to_features_dict()
        
        # Perform inference
        score, is_anomaly, contributions = inference_engine.score(features)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Track metrics
        if metrics_collector:
            metrics_collector.record_scoring(
                is_anomaly=is_anomaly,
                score=score,
                processing_time_ms=processing_time_ms
            )
        
        return ScoreResponse(
            device_id=request.device_id,
            event_ts=request.window_end,
            score=score,
            threshold=inference_engine.threshold,
            is_anomaly=is_anomaly,
            contributions=contributions,
            model_version=inference_engine.model_version,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Windowed scoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Windowed scoring failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": inference_engine.model_version,
        "threshold": inference_engine.threshold,
        "feature_names": inference_engine.feature_names,
        "window_size": inference_engine.window_size,
        "scaler_type": inference_engine.scaler_type,
        "loaded_at": inference_engine.loaded_at.isoformat()
    }


@app.post("/admin/reload-model")
async def reload_model():
    """Reload the model (admin endpoint)."""
    global inference_engine
    
    # This would typically require authentication in production
    model_dir = os.getenv('MODEL_DIR', 'models')
    
    try:
        new_engine = ONNXInferenceEngine(
            onnx_model_path=os.path.join(model_dir, 'model.onnx'),
            scaler_path=os.path.join(model_dir, 'scaler.pkl'),
            threshold_path=os.path.join(model_dir, 'threshold.json'),
            feature_info_path=os.path.join(model_dir, 'feature_info.json')
        )
        
        # Replace the global engine
        inference_engine = new_engine
        
        logger.info("Model reloaded successfully")
        return {"status": "success", "message": "Model reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/admin/reset-metrics")
async def reset_metrics():
    """Reset metrics (admin endpoint)."""
    global metrics_collector
    
    if metrics_collector:
        metrics_collector.reset()
        return {"status": "success", "message": "Metrics reset successfully"}
    else:
        raise HTTPException(status_code=503, detail="Metrics collector not available")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "IoT Anomaly Detection",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": time.time() - app_start_time,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "score": "/score",
            "batch_score": "/score/batch",
            "windowed_score": "/score/windowed",
            "model_info": "/model/info"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    uvicorn.run(
        "src.serving.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
