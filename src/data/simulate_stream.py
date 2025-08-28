"""
IoT data stream simulator for testing and development.
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import os
from pathlib import Path
import argparse

# Azure Event Hub imports (will be optional for development)
try:
    from azure.eventhub.aio import EventHubProducerClient
    from azure.eventhub import EventData
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure Event Hub client not available. Using file output mode.")

logger = logging.getLogger(__name__)


class IoTDataSimulator:
    """Simulates IoT sensor data streams with realistic patterns and anomalies."""
    
    def __init__(self,
                 devices: list = None,
                 base_temp: float = 22.0,
                 temp_variation: float = 2.0,
                 base_humidity: float = 45.0,
                 humidity_variation: float = 5.0,
                 anomaly_probability: float = 0.01,
                 burst_anomaly_probability: float = 0.005,
                 send_interval_ms: int = 100):
        """
        Initialize IoT data simulator.
        
        Args:
            devices: List of device IDs
            base_temp: Base temperature value
            temp_variation: Temperature variation range
            base_humidity: Base humidity value  
            humidity_variation: Humidity variation range
            anomaly_probability: Probability of single anomaly
            burst_anomaly_probability: Probability of anomaly burst
            send_interval_ms: Interval between messages in milliseconds
        """
        self.devices = devices or [f"dev-{i}" for i in range(1, 4)]
        self.base_temp = base_temp
        self.temp_variation = temp_variation
        self.base_humidity = base_humidity
        self.humidity_variation = humidity_variation
        self.anomaly_probability = anomaly_probability
        self.burst_anomaly_probability = burst_anomaly_probability
        self.send_interval_ms = send_interval_ms
        
        # State for each device
        self.device_states = {}
        for device_id in self.devices:
            self.device_states[device_id] = {
                'temp_drift': 0.0,
                'humidity_drift': 0.0,
                'anomaly_burst_remaining': 0,
                'last_temp': self.base_temp,
                'last_humidity': self.base_humidity
            }
        
        # Time tracking
        self.start_time = time.time()
        self.message_count = 0
        
        logger.info(f"Initialized IoT simulator with {len(self.devices)} devices")
    
    def generate_normal_reading(self, device_id: str, timestamp: float) -> Dict:
        """Generate normal sensor reading with daily patterns."""
        state = self.device_states[device_id]
        
        # Time-based patterns (daily cycle)
        hours_since_start = (timestamp - self.start_time) / 3600
        daily_cycle = np.sin(2 * np.pi * hours_since_start / 24)  # 24-hour cycle
        
        # Base values with daily variation
        temp_base = self.base_temp + 3 * daily_cycle + state['temp_drift']
        humidity_base = self.base_humidity - 5 * daily_cycle + state['humidity_drift']
        
        # Add noise
        temp_noise = np.random.normal(0, 0.5)
        humidity_noise = np.random.normal(0, 1.0)
        
        # Smooth transitions (avoid sudden jumps)
        alpha = 0.7  # Smoothing factor
        new_temp = alpha * state['last_temp'] + (1 - alpha) * (temp_base + temp_noise)
        new_humidity = alpha * state['last_humidity'] + (1 - alpha) * (humidity_base + humidity_noise)
        
        # Update state
        state['last_temp'] = new_temp
        state['last_humidity'] = new_humidity
        
        # Gradual drift (simulate sensor aging)
        if np.random.random() < 0.001:  # 0.1% chance of drift change
            state['temp_drift'] += np.random.normal(0, 0.1)
            state['humidity_drift'] += np.random.normal(0, 0.2)
        
        return {
            'deviceId': device_id,
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'temperature': round(float(new_temp), 2),
            'humidity': round(float(new_humidity), 2),
            'enqueuedTime': timestamp
        }
    
    def inject_anomaly(self, reading: Dict, anomaly_type: str = None) -> Dict:
        """Inject anomaly into a normal reading."""
        if anomaly_type is None:
            anomaly_type = np.random.choice(['spike', 'drop', 'offset', 'noise'])
        
        if anomaly_type == 'spike':
            # Temperature spike
            reading['temperature'] += np.random.uniform(8, 15)
        elif anomaly_type == 'drop':
            # Temperature drop
            reading['temperature'] -= np.random.uniform(8, 12)
        elif anomaly_type == 'offset':
            # Sustained offset
            offset = np.random.uniform(5, 10)
            reading['temperature'] += offset
            reading['humidity'] += offset * 0.5
        elif anomaly_type == 'noise':
            # High noise
            reading['temperature'] += np.random.normal(0, 3)
            reading['humidity'] += np.random.normal(0, 5)
        
        # Ensure reasonable bounds
        reading['temperature'] = max(-10, min(60, reading['temperature']))
        reading['humidity'] = max(0, min(100, reading['humidity']))
        
        return reading
    
    def generate_reading(self, device_id: str, timestamp: float) -> Dict:
        """Generate a sensor reading (normal or anomalous)."""
        state = self.device_states[device_id]
        
        # Generate base reading
        reading = self.generate_normal_reading(device_id, timestamp)
        
        # Check for anomaly burst continuation
        if state['anomaly_burst_remaining'] > 0:
            reading = self.inject_anomaly(reading, 'offset')
            state['anomaly_burst_remaining'] -= 1
            reading['anomaly_type'] = 'burst_continuation'
        
        # Check for new anomaly burst
        elif np.random.random() < self.burst_anomaly_probability:
            reading = self.inject_anomaly(reading, 'offset')
            state['anomaly_burst_remaining'] = np.random.randint(5, 15)  # Burst length
            reading['anomaly_type'] = 'burst_start'
        
        # Check for single anomaly
        elif np.random.random() < self.anomaly_probability:
            reading = self.inject_anomaly(reading)
            reading['anomaly_type'] = 'single'
        
        else:
            reading['anomaly_type'] = 'normal'
        
        return reading
    
    async def run_azure_eventhub_producer(self, 
                                        connection_string: str, 
                                        eventhub_name: str,
                                        duration_minutes: int = 60) -> None:
        """Run simulator with Azure Event Hub output."""
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure Event Hub client not available")
        
        producer = EventHubProducerClient.from_connection_string(
            connection_string, eventhub_name=eventhub_name
        )
        
        logger.info(f"Starting Azure Event Hub producer for {duration_minutes} minutes")
        
        async with producer:
            end_time = time.time() + (duration_minutes * 60)
            
            while time.time() < end_time:
                batch_start = time.time()
                
                # Create batch of events
                batch = []
                for device_id in self.devices:
                    reading = self.generate_reading(device_id, batch_start)
                    event_data = EventData(json.dumps(reading))
                    batch.append(event_data)
                    self.message_count += 1
                
                # Send batch
                try:
                    await producer.send_batch(batch)
                    if self.message_count % 100 == 0:
                        logger.info(f"Sent {self.message_count} messages")
                
                except Exception as e:
                    logger.error(f"Failed to send batch: {e}")
                
                # Wait for next interval
                elapsed = (time.time() - batch_start) * 1000
                sleep_time = max(0, (self.send_interval_ms - elapsed) / 1000)
                await asyncio.sleep(sleep_time)
        
        logger.info(f"Simulation completed. Total messages sent: {self.message_count}")
    
    def run_file_output(self, 
                       output_file: str = "simulated_data.jsonl",
                       duration_minutes: int = 10) -> None:
        """Run simulator with file output (for development)."""
        logger.info(f"Starting file output simulation for {duration_minutes} minutes")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        end_time = time.time() + (duration_minutes * 60)
        
        with open(output_path, 'w') as f:
            while time.time() < end_time:
                batch_start = time.time()
                
                # Generate readings for all devices
                for device_id in self.devices:
                    reading = self.generate_reading(device_id, batch_start)
                    f.write(json.dumps(reading) + '\n')
                    self.message_count += 1
                
                # Log progress
                if self.message_count % 100 == 0:
                    logger.info(f"Generated {self.message_count} readings")
                
                # Wait for next interval
                elapsed = (time.time() - batch_start) * 1000
                sleep_time = max(0, (self.send_interval_ms - elapsed) / 1000)
                time.sleep(sleep_time)
        
        logger.info(f"Simulation completed. Total readings: {self.message_count}")
        logger.info(f"Data saved to: {output_path}")
    
    def generate_batch_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate batch data for testing."""
        logger.info(f"Generating batch of {num_samples} samples")
        
        readings = []
        start_time = time.time()
        
        for i in range(num_samples):
            timestamp = start_time + i * (self.send_interval_ms / 1000)
            device_id = self.devices[i % len(self.devices)]
            reading = self.generate_reading(device_id, timestamp)
            readings.append(reading)
        
        df = pd.DataFrame(readings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add some statistics
        anomaly_count = len(df[df['anomaly_type'] != 'normal'])
        logger.info(f"Generated {num_samples} samples with {anomaly_count} anomalies "
                   f"({anomaly_count/num_samples*100:.1f}%)")
        
        return df


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="IoT Data Stream Simulator")
    parser.add_argument("--mode", choices=['eventhub', 'file', 'batch'], 
                       default='file', help="Output mode")
    parser.add_argument("--duration", type=int, default=10, 
                       help="Duration in minutes (for streaming modes)")
    parser.add_argument("--output", default="data/simulated_data.jsonl", 
                       help="Output file path")
    parser.add_argument("--devices", nargs='+', 
                       default=['dev-1', 'dev-2', 'dev-3'], 
                       help="Device IDs")
    parser.add_argument("--interval", type=int, default=100, 
                       help="Send interval in milliseconds")
    
    # Azure-specific arguments
    parser.add_argument("--connection-string", help="Azure Event Hub connection string")
    parser.add_argument("--eventhub-name", help="Event Hub name")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create simulator
    simulator = IoTDataSimulator(
        devices=args.devices,
        send_interval_ms=args.interval
    )
    
    if args.mode == 'eventhub':
        if not args.connection_string or not args.eventhub_name:
            raise ValueError("Azure Event Hub connection string and name are required")
        
        await simulator.run_azure_eventhub_producer(
            args.connection_string,
            args.eventhub_name,
            args.duration
        )
    
    elif args.mode == 'file':
        simulator.run_file_output(args.output, args.duration)
    
    elif args.mode == 'batch':
        df = simulator.generate_batch_data(1000)
        output_path = Path(args.output).with_suffix('.csv')
        df.to_csv(output_path, index=False)
        print(f"Batch data saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
