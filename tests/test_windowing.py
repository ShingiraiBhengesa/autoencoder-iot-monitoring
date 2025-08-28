"""
Tests for windowing functionality.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.windowing import (
    make_windows, make_windows_with_labels, 
    TimeSeriesWindower, prepare_train_val_test
)


class TestMakeWindows:
    """Test the make_windows function."""
    
    def test_basic_windowing(self):
        """Test basic windowing functionality."""
        # Create sample data
        data = np.random.randn(100, 2)
        windows = make_windows(data, win=10, stride=1)
        
        assert windows.shape == (91, 10, 2)
        assert np.array_equal(windows[0], data[0:10])
        assert np.array_equal(windows[1], data[1:11])
    
    def test_stride_greater_than_1(self):
        """Test windowing with stride > 1."""
        data = np.random.randn(50, 3)
        windows = make_windows(data, win=5, stride=2)
        
        expected_n_windows = 1 + (50 - 5) // 2
        assert windows.shape == (expected_n_windows, 5, 3)
    
    def test_insufficient_data(self):
        """Test error when data is smaller than window."""
        data = np.random.randn(5, 2)
        with pytest.raises(ValueError):
            make_windows(data, win=10, stride=1)
    
    def test_exact_window_size(self):
        """Test when data length equals window size."""
        data = np.random.randn(10, 2)
        windows = make_windows(data, win=10, stride=1)
        
        assert windows.shape == (1, 10, 2)
        assert np.array_equal(windows[0], data)


class TestMakeWindowsWithLabels:
    """Test windowing with labels."""
    
    def test_windows_with_labels(self):
        """Test windowing with labels."""
        data = np.random.randn(20, 2)
        labels = np.random.randint(0, 2, 20)
        
        windowed_data, windowed_labels = make_windows_with_labels(
            data, labels, win=5, stride=1
        )
        
        assert windowed_data.shape == (16, 5, 2)
        assert windowed_labels.shape == (16,)
        
        # Check that labels correspond to end of windows
        assert windowed_labels[0] == labels[4]  # End of first window
        assert windowed_labels[1] == labels[5]  # End of second window
    
    def test_no_labels(self):
        """Test windowing without labels."""
        data = np.random.randn(15, 2)
        windowed_data, windowed_labels = make_windows_with_labels(
            data, None, win=3, stride=1
        )
        
        assert windowed_data.shape == (13, 3, 2)
        assert windowed_labels is None


class TestTimeSeriesWindower:
    """Test TimeSeriesWindower class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'temperature': 20 + 5 * np.sin(np.arange(100) / 10) + np.random.normal(0, 0.5, 100),
            'humidity': 50 + 10 * np.cos(np.arange(100) / 10) + np.random.normal(0, 1, 100)
        }, index=dates)
        return data
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        labels = pd.Series(np.random.choice([0, 1], 100, p=[0.9, 0.1]), index=dates)
        return labels
    
    def test_fit_transform(self, sample_data, sample_labels):
        """Test fit and transform."""
        windower = TimeSeriesWindower(window_size=10, stride=1)
        windowed_data, windowed_labels, timestamps = windower.fit_transform(
            sample_data, sample_labels
        )
        
        expected_n_windows = 1 + (100 - 10) // 1
        assert windowed_data.shape == (expected_n_windows, 10 * 2)  # Flattened
        assert windowed_labels.shape == (expected_n_windows,)
        assert len(timestamps) == expected_n_windows
        
        # Check scaler was fitted
        assert windower.scaler is not None
        assert windower.feature_names == ['temperature', 'humidity']
    
    def test_transform_without_labels(self, sample_data):
        """Test transform without labels."""
        windower = TimeSeriesWindower(window_size=5, stride=2)
        windowed_data, windowed_labels, timestamps = windower.fit_transform(sample_data)
        
        expected_n_windows = 1 + (100 - 5) // 2
        assert windowed_data.shape == (expected_n_windows, 5 * 2)
        assert windowed_labels is None
        assert len(timestamps) == expected_n_windows
    
    def test_scaler_types(self, sample_data):
        """Test different scaler types."""
        for scaler_type in ["MinMaxScaler", "RobustScaler"]:
            windower = TimeSeriesWindower(scaler_type=scaler_type)
            windower.fit(sample_data)
            
            assert windower.scaler_type == scaler_type
            assert windower.scaler is not None


class TestPrepareTrainValTest:
    """Test prepare_train_val_test function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for splitting."""
        dates = pd.date_range('2024-01-01', periods=200, freq='30T')
        data = pd.DataFrame({
            'temp': 20 + np.random.normal(0, 2, 200),
            'humidity': 50 + np.random.normal(0, 5, 200)
        }, index=dates)
        labels = pd.Series(np.random.choice([0, 1], 200, p=[0.95, 0.05]), index=dates)
        return data, labels
    
    def test_basic_split(self, sample_data):
        """Test basic train/val/test split."""
        data, labels = sample_data
        
        datasets = prepare_train_val_test(
            data, labels,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            window_size=5, stride=1
        )
        
        # Check structure
        assert 'windower' in datasets
        assert 'train' in datasets
        assert 'val' in datasets
        assert 'test' in datasets
        
        # Check each split has required keys
        for split in ['train', 'val', 'test']:
            assert 'X' in datasets[split]
            assert 'X_3d' in datasets[split]
            assert 'y' in datasets[split]
            assert 'timestamps' in datasets[split]
            assert 'raw_data' in datasets[split]
        
        # Check windower
        assert datasets['windower'].window_size == 5
        assert datasets['windower'].stride == 1
    
    def test_no_labels_split(self, sample_data):
        """Test split without labels."""
        data, _ = sample_data
        
        datasets = prepare_train_val_test(
            data, labels=None,
            window_size=10, stride=2
        )
        
        # Should still work without labels
        assert datasets['train']['y'] is None
        assert datasets['val']['y'] is None
        assert datasets['test']['y'] is None
    
    def test_ratios_validation(self, sample_data):
        """Test ratio validation."""
        data, labels = sample_data
        
        with pytest.raises(AssertionError):
            prepare_train_val_test(
                data, labels,
                train_ratio=0.5, val_ratio=0.3, test_ratio=0.3  # Sum > 1
            )
    
    def test_time_based_split(self, sample_data):
        """Test that split is time-based (no data leakage)."""
        data, labels = sample_data
        
        datasets = prepare_train_val_test(
            data, labels,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        # Check temporal ordering
        train_end = datasets['train']['timestamps'][-1]
        val_start = datasets['val']['timestamps'][0]
        val_end = datasets['val']['timestamps'][-1]
        test_start = datasets['test']['timestamps'][0]
        
        assert train_end <= val_start
        assert val_end <= test_start


if __name__ == "__main__":
    pytest.main([__file__])
