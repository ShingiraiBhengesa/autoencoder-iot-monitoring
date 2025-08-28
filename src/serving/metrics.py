"""
Metrics collection for the anomaly detection service.
"""

import time
import threading
from typing import Dict, List, Optional
from collections import deque, defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collector for service monitoring."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of recent measurements to keep
        """
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Request metrics
        self.total_requests = 0
        self.request_times = deque(maxlen=max_history)
        self.request_timestamps = deque(maxlen=max_history)
        self.status_codes = defaultdict(int)
        self.endpoints = defaultdict(int)
        
        # Scoring metrics
        self.total_scores = 0
        self.total_anomalies = 0
        self.scoring_times = deque(maxlen=max_history)
        self.anomaly_scores = deque(maxlen=max_history)
        
        # Performance tracking
        self.start_time = time.time()
        
        logger.info("Metrics collector initialized")
    
    def record_request(self, 
                      endpoint: str,
                      method: str,
                      status_code: int,
                      duration_ms: float) -> None:
        """
        Record a request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
        """
        with self._lock:
            self.total_requests += 1
            self.request_times.append(duration_ms)
            self.request_timestamps.append(time.time())
            self.status_codes[status_code] += 1
            self.endpoints[f"{method} {endpoint}"] += 1
    
    def record_scoring(self,
                      is_anomaly: bool,
                      score: float,
                      processing_time_ms: float) -> None:
        """
        Record a scoring operation.
        
        Args:
            is_anomaly: Whether the sample was classified as anomalous
            score: Anomaly score
            processing_time_ms: Processing time in milliseconds
        """
        with self._lock:
            self.total_scores += 1
            if is_anomaly:
                self.total_anomalies += 1
            
            self.scoring_times.append(processing_time_ms)
            self.anomaly_scores.append(score)
    
    def get_metrics(self) -> Dict:
        """
        Get comprehensive metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            current_time = time.time()
            
            # Request metrics
            if self.request_times:
                avg_processing_time = statistics.mean(self.request_times)
                p50_processing_time = statistics.median(self.request_times)
                p95_processing_time = self._percentile(self.request_times, 95)
                p99_processing_time = self._percentile(self.request_times, 99)
            else:
                avg_processing_time = 0.0
                p50_processing_time = 0.0
                p95_processing_time = 0.0
                p99_processing_time = 0.0
            
            # Anomaly rate
            anomaly_rate = self.total_anomalies / self.total_scores if self.total_scores > 0 else 0.0
            
            # Requests per second (last minute)
            recent_requests = self._count_recent_requests(60)
            requests_per_second = recent_requests / 60.0
            
            return {
                'total_requests': self.total_requests,
                'total_anomalies': self.total_anomalies,
                'avg_processing_time_ms': avg_processing_time,
                'p50_processing_time_ms': p50_processing_time,
                'p95_processing_time_ms': p95_processing_time,
                'p99_processing_time_ms': p99_processing_time,
                'anomaly_rate': anomaly_rate,
                'requests_per_second': requests_per_second,
                'status_code_distribution': dict(self.status_codes),
                'endpoint_distribution': dict(self.endpoints),
                'uptime_seconds': current_time - self.start_time
            }
    
    def get_scoring_metrics(self) -> Dict:
        """
        Get detailed scoring metrics.
        
        Returns:
            Dictionary of scoring-specific metrics
        """
        with self._lock:
            if not self.scoring_times:
                return {
                    'total_scores': 0,
                    'total_anomalies': 0,
                    'anomaly_rate': 0.0,
                    'avg_score': 0.0,
                    'avg_scoring_time_ms': 0.0,
                    'score_distribution': {}
                }
            
            # Score statistics
            avg_score = statistics.mean(self.anomaly_scores)
            median_score = statistics.median(self.anomaly_scores)
            max_score = max(self.anomaly_scores)
            min_score = min(self.anomaly_scores)
            
            # Score distribution (binned)
            score_bins = self._create_score_distribution()
            
            # Timing statistics
            avg_scoring_time = statistics.mean(self.scoring_times)
            
            return {
                'total_scores': self.total_scores,
                'total_anomalies': self.total_anomalies,
                'anomaly_rate': self.total_anomalies / self.total_scores,
                'avg_score': avg_score,
                'median_score': median_score,
                'min_score': min_score,
                'max_score': max_score,
                'avg_scoring_time_ms': avg_scoring_time,
                'score_distribution': score_bins
            }
    
    def get_recent_performance(self, minutes: int = 5) -> Dict:
        """
        Get performance metrics for recent time window.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            Recent performance metrics
        """
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)
            
            # Filter recent requests
            recent_indices = [
                i for i, ts in enumerate(self.request_timestamps)
                if ts > cutoff_time
            ]
            
            if not recent_indices:
                return {
                    'time_window_minutes': minutes,
                    'requests': 0,
                    'requests_per_second': 0.0,
                    'avg_processing_time_ms': 0.0,
                    'error_rate': 0.0
                }
            
            recent_times = [self.request_times[i] for i in recent_indices]
            avg_time = statistics.mean(recent_times)
            
            # Count errors (4xx, 5xx)
            recent_errors = sum(
                1 for code, count in self.status_codes.items()
                if code >= 400
            )
            error_rate = recent_errors / len(recent_indices) if recent_indices else 0.0
            
            return {
                'time_window_minutes': minutes,
                'requests': len(recent_indices),
                'requests_per_second': len(recent_indices) / (minutes * 60),
                'avg_processing_time_ms': avg_time,
                'error_rate': error_rate
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_requests = 0
            self.total_scores = 0
            self.total_anomalies = 0
            
            self.request_times.clear()
            self.request_timestamps.clear()
            self.scoring_times.clear()
            self.anomaly_scores.clear()
            
            self.status_codes.clear()
            self.endpoints.clear()
            
            self.start_time = time.time()
            
        logger.info("Metrics reset")
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _count_recent_requests(self, seconds: int) -> int:
        """Count requests in the last N seconds."""
        cutoff_time = time.time() - seconds
        return sum(1 for ts in self.request_timestamps if ts > cutoff_time)
    
    def _create_score_distribution(self) -> Dict[str, int]:
        """Create binned distribution of anomaly scores."""
        if not self.anomaly_scores:
            return {}
        
        # Define bins
        bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf')]
        bin_labels = [
            '0.00-0.01', '0.01-0.05', '0.05-0.10', 
            '0.10-0.20', '0.20-0.50', '0.50-1.00', '1.00+'
        ]
        
        distribution = {label: 0 for label in bin_labels}
        
        for score in self.anomaly_scores:
            for i, threshold in enumerate(bins[1:], 0):
                if score <= threshold:
                    distribution[bin_labels[i]] += 1
                    break
        
        return distribution


class MetricsExporter:
    """Export metrics in various formats."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.metrics_collector.get_metrics()
        scoring_metrics = self.metrics_collector.get_scoring_metrics()
        
        prometheus_output = []
        
        # Request metrics
        prometheus_output.extend([
            "# HELP iot_anomaly_requests_total Total number of requests",
            "# TYPE iot_anomaly_requests_total counter",
            f"iot_anomaly_requests_total {metrics['total_requests']}",
            "",
            "# HELP iot_anomaly_processing_time_seconds Request processing time",
            "# TYPE iot_anomaly_processing_time_seconds summary",
            f"iot_anomaly_processing_time_seconds {{quantile=\"0.5\"}} {metrics['p50_processing_time_ms']/1000}",
            f"iot_anomaly_processing_time_seconds {{quantile=\"0.95\"}} {metrics['p95_processing_time_ms']/1000}",
            f"iot_anomaly_processing_time_seconds {{quantile=\"0.99\"}} {metrics['p99_processing_time_ms']/1000}",
            "",
            "# HELP iot_anomaly_score_total Total anomalies detected",
            "# TYPE iot_anomaly_score_total counter", 
            f"iot_anomaly_score_total {metrics['total_anomalies']}",
            "",
            "# HELP iot_anomaly_rate Anomaly detection rate",
            "# TYPE iot_anomaly_rate gauge",
            f"iot_anomaly_rate {metrics['anomaly_rate']}",
        ])
        
        return "\n".join(prometheus_output)
    
    def to_json(self) -> Dict:
        """Export metrics as JSON."""
        return self.metrics_collector.get_metrics()


if __name__ == "__main__":
    # Test metrics collector
    collector = MetricsCollector()
    
    # Simulate some requests
    collector.record_request("/score", "POST", 200, 15.5)
    collector.record_request("/health", "GET", 200, 2.1)
    collector.record_request("/score", "POST", 200, 18.3)
    
    # Simulate scoring
    collector.record_scoring(True, 0.8, 12.5)  # Anomaly
    collector.record_scoring(False, 0.02, 10.1)  # Normal
    collector.record_scoring(False, 0.05, 11.8)  # Normal
    
    # Get metrics
    metrics = collector.get_metrics()
    print("Metrics:", metrics)
    
    # Test exporter
    exporter = MetricsExporter(collector)
    print("\nPrometheus format:")
    print(exporter.to_prometheus())
