"""
Sliding window implementation for time series data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


def make_windows(arr: np.ndarray, win: int = 30, stride: int = 1) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        arr: Input array of shape (T, F) where T is time steps, F is features
        win: Window size
        stride: Stride between windows
    
    Returns:
        Windows of shape (n_windows, win, F)
    """
    T, F = arr.shape
    if T < win:
        raise ValueError(f"Array length {T} is smaller than window size {win}")
    
    n = 1 + (T - win) // stride
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return arr[idx]  # (n, win, F)


def make_windows_with_labels(data: np.ndarray, 
                           labels: Optional[np.ndarray], 
                           win: int = 30,
                           stride: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create sliding windows for both data and labels.
    
    Args:
        data: Input data array of shape (T, F)
        labels: Labels array of shape (T,) or None
        win: Window size
        stride: Stride between windows
    
    Returns:
        Tuple of (windowed_data, windowed_labels)
    """
    windowed_data = make_windows(data, win, stride)
    
    if labels is not None:
        # For labels, take the label at the end of each window
        T = len(labels)
        n = 1 + (T - win) // stride
        label_indices = win - 1 + stride * np.arange(n)
        windowed_labels = labels[label_indices]
        return windowed_data, windowed_labels
    
    return windowed_data, None


class TimeSeriesWindower:
    """Time series windowing with preprocessing."""
    
    def __init__(self, 
                 window_size: int = 30,
                 stride: int = 1,
                 scaler_type: str = "MinMaxScaler"):
        self.window_size = window_size
        self.stride = stride
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        
    def _get_scaler(self):
        """Get scaler instance."""
        if self.scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        elif self.scaler_type == "RobustScaler":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesWindower':
        """
        Fit the windower on training data.
        
        Args:
            data: DataFrame with time series data
        
        Returns:
            Self for chaining
        """
        self.feature_names = data.columns.tolist()
        
        # Fit scaler on entire dataset
        self.scaler = self._get_scaler()
        self.scaler.fit(data.values)
        
        logger.info(f"Fitted windower: {len(self.feature_names)} features, "
                   f"window_size={self.window_size}, scaler={self.scaler_type}")
        
        return self
    
    def transform(self, data: pd.DataFrame, 
                  labels: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Transform data into windowed format.
        
        Args:
            data: DataFrame with time series data
            labels: Optional labels series
        
        Returns:
            Tuple of (windowed_data, windowed_labels, timestamps)
        """
        if self.scaler is None:
            raise ValueError("Windower not fitted. Call fit() first.")
        
        # Scale data
        scaled_data = self.scaler.transform(data.values)
        
        # Create windows
        windowed_data, windowed_labels = make_windows_with_labels(
            scaled_data, 
            labels.values if labels is not None else None,
            self.window_size, 
            self.stride
        )
        
        # Get timestamps for each window (use end timestamp)
        T = len(data)
        n = 1 + (T - self.window_size) // self.stride
        timestamp_indices = self.window_size - 1 + self.stride * np.arange(n)
        timestamps = data.index[timestamp_indices].values
        
        logger.info(f"Created {len(windowed_data)} windows of size {self.window_size}")
        
        return windowed_data, windowed_labels, timestamps
    
    def fit_transform(self, data: pd.DataFrame, 
                     labels: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data, labels)


def prepare_train_val_test(data: pd.DataFrame,
                          labels: Optional[pd.Series] = None,
                          train_ratio: float = 0.6,
                          val_ratio: float = 0.2,
                          test_ratio: float = 0.2,
                          window_size: int = 30,
                          stride: int = 1,
                          scaler_type: str = "MinMaxScaler") -> dict:
    """
    Prepare train/validation/test sets with proper windowing.
    
    Args:
        data: Input DataFrame
        labels: Optional labels
        train_ratio: Training data ratio
        val_ratio: Validation data ratio  
        test_ratio: Test data ratio
        window_size: Window size for sliding windows
        stride: Stride between windows
        scaler_type: Type of scaler to use
    
    Returns:
        Dictionary with train/val/test data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Time-based split (no data leakage)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    train_labels = labels.iloc[:train_end] if labels is not None else None
    val_labels = labels.iloc[train_end:val_end] if labels is not None else None
    test_labels = labels.iloc[val_end:] if labels is not None else None
    
    # Create windower and fit on training data only
    windower = TimeSeriesWindower(window_size, stride, scaler_type)
    windower.fit(train_data)
    
    # Transform all sets
    train_windows, train_window_labels, train_timestamps = windower.transform(train_data, train_labels)
    val_windows, val_window_labels, val_timestamps = windower.transform(val_data, val_labels)
    test_windows, test_window_labels, test_timestamps = windower.transform(test_data, test_labels)
    
    # Flatten windows for dense autoencoder (keep 3D for LSTM)
    train_windows_flat = train_windows.reshape(train_windows.shape[0], -1)
    val_windows_flat = val_windows.reshape(val_windows.shape[0], -1)
    test_windows_flat = test_windows.reshape(test_windows.shape[0], -1)
    
    result = {
        'windower': windower,
        'train': {
            'X': train_windows_flat,
            'X_3d': train_windows,
            'y': train_window_labels,
            'timestamps': train_timestamps,
            'raw_data': train_data
        },
        'val': {
            'X': val_windows_flat,
            'X_3d': val_windows,
            'y': val_window_labels,
            'timestamps': val_timestamps,
            'raw_data': val_data
        },
        'test': {
            'X': test_windows_flat,
            'X_3d': test_windows,
            'y': test_window_labels,
            'timestamps': test_timestamps,
            'raw_data': test_data
        }
    }
    
    logger.info(f"Prepared datasets:")
    logger.info(f"  Train: {train_windows_flat.shape[0]} windows")
    logger.info(f"  Val: {val_windows_flat.shape[0]} windows") 
    logger.info(f"  Test: {test_windows_flat.shape[0]} windows")
    logger.info(f"  Window shape (flat): {train_windows_flat.shape[1:]}")
    logger.info(f"  Window shape (3D): {train_windows.shape[1:]}")
    
    return result


if __name__ == "__main__":
    # Test windowing functionality
    np.random.seed(42)
    
    # Create sample data
    n_points = 1000
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1H')
    temp = 20 + 5 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 0.5, n_points)
    humidity = 50 + 10 * np.cos(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 1, n_points)
    
    data = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity
    }, index=timestamps)
    
    # Create some artificial labels
    labels = pd.Series(np.random.choice([0, 1], n_points, p=[0.9, 0.1]), index=timestamps)
    
    # Test windowing
    datasets = prepare_train_val_test(data, labels, window_size=24, stride=1)
    
    print("Windowing test complete!")
    print(f"Train X shape: {datasets['train']['X'].shape}")
    print(f"Val X shape: {datasets['val']['X'].shape}")
    print(f"Test X shape: {datasets['test']['X'].shape}")
