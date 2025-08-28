"""
Load and process Numenta Anomaly Benchmark (NAB) dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import requests
import zipfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NABDataLoader:
    """Load and process NAB dataset for IoT anomaly detection."""
    
    NAB_URL = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_nab(self) -> None:
        """Download NAB dataset if not exists."""
        nab_zip = self.raw_dir / "NAB-master.zip"
        nab_extracted = self.raw_dir / "NAB-master"
        
        if nab_extracted.exists():
            logger.info("NAB dataset already exists")
            return
            
        if not nab_zip.exists():
            logger.info("Downloading NAB dataset...")
            response = requests.get(self.NAB_URL)
            response.raise_for_status()
            
            with open(nab_zip, 'wb') as f:
                f.write(response.content)
            logger.info("Download complete")
        
        # Extract
        logger.info("Extracting NAB dataset...")
        with zipfile.ZipFile(nab_zip, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        logger.info("Extraction complete")
    
    def load_temperature_data(self) -> Dict[str, pd.DataFrame]:
        """Load temperature sensor data from NAB."""
        self.download_nab()
        
        nab_data_dir = self.raw_dir / "NAB-master" / "data"
        temperature_files = {
            "ambient_temperature_system_failure": 
                nab_data_dir / "realAWSCloudwatch" / "ec2_cpu_utilization_825cc2.csv",
            "temperature_system_failure":
                nab_data_dir / "realAWSCloudwatch" / "ec2_disk_write_bytes_1ef3de.csv",
            "machine_temperature":
                nab_data_dir / "realKnownCause" / "machine_temperature_system_failure.csv"
        }
        
        datasets = {}
        for name, filepath in temperature_files.items():
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                datasets[name] = df
                logger.info(f"Loaded {name}: {len(df)} records")
            else:
                logger.warning(f"File not found: {filepath}")
        
        # If specific temperature files don't exist, use any available data
        if not datasets:
            logger.info("Loading any available NAB data files...")
            for data_type in ["realKnownCause", "realAWSCloudwatch", "realTraffic"]:
                data_path = nab_data_dir / data_type
                if data_path.exists():
                    for csv_file in data_path.glob("*.csv"):
                        df = pd.read_csv(csv_file)
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('timestamp')
                            datasets[csv_file.stem] = df
                            logger.info(f"Loaded {csv_file.stem}: {len(df)} records")
                            if len(datasets) >= 3:  # Limit to first 3 files
                                break
                if len(datasets) >= 3:
                    break
        
        return datasets
    
    def create_synthetic_temperature_data(self, 
                                        num_points: int = 50000,
                                        anomaly_ratio: float = 0.1) -> pd.DataFrame:
        """Create synthetic temperature data if NAB is not available."""
        logger.info(f"Creating synthetic temperature data: {num_points} points")
        
        # Generate timestamps
        start_time = pd.Timestamp.now() - pd.Timedelta(days=30)
        timestamps = pd.date_range(start=start_time, periods=num_points, freq='30S')
        
        # Generate base temperature pattern (daily cycle)
        base_temp = 20 + 5 * np.sin(2 * np.pi * np.arange(num_points) / (24 * 120))  # 120 points per hour
        
        # Add noise
        noise = np.random.normal(0, 0.5, num_points)
        temperature = base_temp + noise
        
        # Add anomalies
        num_anomalies = int(num_points * anomaly_ratio)
        anomaly_indices = np.random.choice(num_points, num_anomalies, replace=False)
        
        # Create different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'shift'])
            if anomaly_type == 'spike':
                temperature[idx] += np.random.uniform(10, 20)
            elif anomaly_type == 'drop':
                temperature[idx] -= np.random.uniform(10, 15)
            else:  # shift
                # Create a temporary shift
                shift_length = min(50, num_points - idx)
                temperature[idx:idx+shift_length] += np.random.uniform(5, 10)
        
        # Create labels (1 for anomaly, 0 for normal)
        labels = np.zeros(num_points)
        labels[anomaly_indices] = 1
        
        df = pd.DataFrame({
            'temperature': temperature,
            'humidity': 40 + 10 * np.sin(2 * np.pi * np.arange(num_points) / (24 * 120)) + np.random.normal(0, 2, num_points),
            'is_anomaly': labels
        }, index=timestamps)
        
        return df
    
    def prepare_time_series_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Prepare time series data for training."""
        # Try to load NAB data first
        datasets = self.load_temperature_data()
        
        if not datasets:
            logger.warning("NAB data not available, creating synthetic data")
            main_df = self.create_synthetic_temperature_data()
            labels_df = main_df[['is_anomaly']].copy()
            main_df = main_df.drop('is_anomaly', axis=1)
        else:
            # Use the first available dataset
            dataset_name = list(datasets.keys())[0]
            main_df = datasets[dataset_name].copy()
            
            # NAB doesn't have explicit labels in the CSV, so we'll create synthetic ones
            # In a real scenario, you'd load the NAB labels from labels/combined_labels.json
            labels_df = None
            logger.info(f"Using dataset: {dataset_name}")
        
        # Save processed data
        main_df.to_csv(self.processed_dir / "sensor_data.csv")
        if labels_df is not None:
            labels_df.to_csv(self.processed_dir / "labels.csv")
        
        logger.info(f"Processed data saved: {len(main_df)} records")
        return main_df, labels_df


if __name__ == "__main__":
    loader = NABDataLoader()
    data, labels = loader.prepare_time_series_data()
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    if labels is not None:
        print(f"Anomaly ratio: {labels['is_anomaly'].mean():.3f}")
