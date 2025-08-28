"""
ONNX inference engine for production anomaly scoring.
"""

import numpy as np
import onnxruntime as ort
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """High-performance ONNX inference engine for anomaly detection."""
    
    def __init__(self,
                 onnx_model_path: str,
                 scaler_path: str,
                 threshold_path: str,
                 feature_info_path: str):
        """
        Initialize ONNX inference engine.
        
        Args:
            onnx_model_path: Path to ONNX model file
            scaler_path: Path to fitted scaler pickle file
            threshold_path: Path to threshold configuration JSON
            feature_info_path: Path to feature information JSON
        """
        self.onnx_model_path = onnx_model_path
        self.scaler_path = scaler_path
        self.threshold_path = threshold_path
        self.feature_info_path = feature_info_path
        
        # Model components
        self.session = None
        self.scaler = None
        self.threshold_config = None
        self.feature_info = None
        
        # Derived properties
        self.threshold = None
        self.model_version = None
        self.feature_names = None
        self.window_size = None
        self.scaler_type = None
        self.loaded_at = None
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Load all components
        self._load_components()
    
    def _load_components(self) -> None:
        """Load all model components."""
        logger.info("Loading ONNX inference engine components...")
        
        try:
            # Load ONNX model
            self._load_onnx_model()
            
            # Load scaler
            self._load_scaler()
            
            # Load threshold configuration
            self._load_threshold_config()
            
            # Load feature information
            self._load_feature_info()
            
            self.loaded_at = datetime.now()
            logger.info("ONNX inference engine loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load inference engine: {e}")
            raise
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        logger.info(f"Loading ONNX model from {self.onnx_model_path}")
        
        # Configure ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1  # Single thread for consistent latency
        
        # Create inference session
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']  # Use CPU for consistent performance
        )
        
        # Log model info
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        
        logger.info(f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        logger.info(f"Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
    
    def _load_scaler(self) -> None:
        """Load fitted scaler."""
        logger.info(f"Loading scaler from {self.scaler_path}")
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Scaler loaded: {type(self.scaler).__name__}")
    
    def _load_threshold_config(self) -> None:
        """Load threshold configuration."""
        logger.info(f"Loading threshold config from {self.threshold_path}")
        
        with open(self.threshold_path, 'r') as f:
            self.threshold_config = json.load(f)
        
        self.threshold = self.threshold_config['threshold']
        logger.info(f"Anomaly threshold: {self.threshold}")
    
    def _load_feature_info(self) -> None:
        """Load feature information."""
        logger.info(f"Loading feature info from {self.feature_info_path}")
        
        with open(self.feature_info_path, 'r') as f:
            self.feature_info = json.load(f)
        
        self.feature_names = self.feature_info['feature_names']
        self.window_size = self.feature_info['window_size']
        self.scaler_type = self.feature_info['scaler_type']
        self.model_version = self.feature_info.get('model_version', 'unknown')
        
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Window size: {self.window_size}")
    
    def _preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Preprocess raw features for model input.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Preprocessed feature array ready for model input
        """
        # For real-time scoring, we simulate a window by repeating current values
        # In production, you'd maintain a sliding window buffer per device
        
        # Extract expected features
        feature_values = []
        for feat_name in self.feature_names:
            if feat_name in features:
                feature_values.append(features[feat_name])
            else:
                # Use default values for missing features
                default_value = 20.0 if 'temp' in feat_name.lower() else 50.0
                feature_values.append(default_value)
                logger.warning(f"Missing feature {feat_name}, using default: {default_value}")
        
        # Create window by repeating current values (simplified approach)
        window_data = np.array([feature_values] * self.window_size)
        
        # Apply scaling
        scaled_window = self.scaler.transform(window_data)
        
        # Flatten for dense autoencoder
        flattened_input = scaled_window.flatten().reshape(1, -1)
        
        return flattened_input.astype(np.float32)
    
    def _run_inference(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference.
        
        Args:
            input_array: Preprocessed input array
        
        Returns:
            Model output (reconstruction)
        """
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        output = self.session.run(None, {input_name: input_array})
        
        return output[0]
    
    def _compute_anomaly_score(self, input_array: np.ndarray, reconstruction: np.ndarray) -> float:
        """
        Compute anomaly score from reconstruction error.
        
        Args:
            input_array: Original input
            reconstruction: Model reconstruction
        
        Returns:
            Anomaly score (MSE)
        """
        # Compute mean squared error
        mse_error = np.mean((input_array - reconstruction) ** 2)
        return float(mse_error)
    
    def _compute_feature_contributions(self, 
                                     input_array: np.ndarray, 
                                     reconstruction: np.ndarray) -> Dict[str, float]:
        """
        Compute per-feature contributions to anomaly score.
        
        Args:
            input_array: Original input
            reconstruction: Model reconstruction
        
        Returns:
            Dictionary of feature contributions
        """
        # Compute per-feature squared errors
        feature_errors = (input_array - reconstruction) ** 2
        
        # Map back to original features
        contributions = {}
        n_features = len(self.feature_names)
        
        for i, feat_name in enumerate(self.feature_names):
            # Average error across the window for this feature
            feat_start = i * self.window_size
            feat_end = (i + 1) * self.window_size
            feat_error = np.mean(feature_errors[0, feat_start:feat_end])
            contributions[feat_name] = float(feat_error)
        
        return contributions
    
    def score(self, features: Dict[str, float]) -> Tuple[float, bool, Dict[str, float]]:
        """
        Score features for anomaly detection.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Tuple of (score, is_anomaly, contributions)
        """
        start_time = datetime.now()
        
        try:
            # Preprocess features
            input_array = self._preprocess_features(features)
            
            # Run inference
            reconstruction = self._run_inference(input_array)
            
            # Compute anomaly score
            score = self._compute_anomaly_score(input_array, reconstruction)
            
            # Determine if anomalous
            is_anomaly = score > self.threshold
            
            # Compute feature contributions
            contributions = self._compute_feature_contributions(input_array, reconstruction)
            
            # Track performance
            inference_time = (datetime.now() - start_time).total_seconds()
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return score, is_anomaly, contributions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise
    
    def batch_score(self, feature_batch: List[Dict[str, float]]) -> List[Tuple[float, bool, Dict[str, float]]]:
        """
        Score a batch of features.
        
        Args:
            feature_batch: List of feature dictionaries
        
        Returns:
            List of (score, is_anomaly, contributions) tuples
        """
        results = []
        
        for features in feature_batch:
            result = self.score(features)
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if self.inference_count == 0:
            return {
                'inference_count': 0,
                'avg_inference_time_ms': 0.0,
                'total_inference_time_s': 0.0
            }
        
        avg_time_ms = (self.total_inference_time / self.inference_count) * 1000
        
        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_time_ms,
            'total_inference_time_s': self.total_inference_time
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.inference_count = 0
        self.total_inference_time = 0.0


# Mock inference engine for development/testing
class MockONNXInferenceEngine:
    """Mock inference engine for development when models aren't available."""
    
    def __init__(self):
        self.threshold = 0.1
        self.model_version = "mock_v1.0"
        self.feature_names = ["temperature", "humidity"]
        self.window_size = 30
        self.scaler_type = "MinMaxScaler"
        self.loaded_at = datetime.now()
        
        logger.info("Mock inference engine initialized")
    
    def score(self, features: Dict[str, float]) -> Tuple[float, bool, Dict[str, float]]:
        """Mock scoring function."""
        # Generate mock score based on features
        temp = features.get('temperature', 20.0)
        humidity = features.get('humidity', 50.0)
        
        # Simple mock scoring logic
        temp_deviation = abs(temp - 22.0) / 10.0
        humidity_deviation = abs(humidity - 45.0) / 20.0
        
        score = temp_deviation + humidity_deviation + np.random.normal(0, 0.01)
        is_anomaly = score > self.threshold
        
        contributions = {
            'temperature': float(temp_deviation),
            'humidity': float(humidity_deviation)
        }
        
        return float(score), is_anomaly, contributions
    
    def batch_score(self, feature_batch: List[Dict[str, float]]) -> List[Tuple[float, bool, Dict[str, float]]]:
        """Mock batch scoring."""
        return [self.score(features) for features in feature_batch]
    
    def get_performance_stats(self) -> Dict:
        """Mock performance stats."""
        return {
            'inference_count': 100,
            'avg_inference_time_ms': 5.0,
            'total_inference_time_s': 0.5
        }


if __name__ == "__main__":
    # Test the inference engine
    import tempfile
    import os
    
    # Test mock engine
    print("Testing Mock Inference Engine:")
    mock_engine = MockONNXInferenceEngine()
    
    test_features = {
        'temperature': 25.5,
        'humidity': 48.2
    }
    
    score, is_anomaly, contributions = mock_engine.score(test_features)
    print(f"Score: {score:.6f}, Anomaly: {is_anomaly}")
    print(f"Contributions: {contributions}")
    
    # Test batch scoring
    batch_features = [
        {'temperature': 22.0, 'humidity': 45.0},
        {'temperature': 35.0, 'humidity': 80.0},  # Anomalous
        {'temperature': 21.5, 'humidity': 42.0}
    ]
    
    batch_results = mock_engine.batch_score(batch_features)
    print(f"\nBatch results:")
    for i, (score, is_anomaly, contrib) in enumerate(batch_results):
        print(f"  Sample {i+1}: score={score:.6f}, anomaly={is_anomaly}")
