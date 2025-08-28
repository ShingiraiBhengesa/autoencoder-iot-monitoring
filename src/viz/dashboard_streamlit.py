"""
Streamlit dashboard for real-time IoT anomaly detection monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlalchemy as sa
from sqlalchemy import text
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="IoT Anomaly Detection Dashboard", 
    page_icon="🌡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .normal-status {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class AnomalyDashboard:
    """Main dashboard class for IoT anomaly detection monitoring."""
    
    def __init__(self):
        self.api_base_url = self._get_api_url()
        self.db_url = self._get_db_url()
        self.engine = None
        
        if self.db_url:
            try:
                self.engine = sa.create_engine(self.db_url)
                logger.info("Database connection established")
            except Exception as e:
                logger.warning(f"Database connection failed: {e}")
    
    def _get_api_url(self) -> str:
        """Get API URL from environment or use default."""
        return os.getenv("API_URL", "http://localhost:8000")
    
    def _get_db_url(self) -> Optional[str]:
        """Get database URL from secrets or environment."""
        try:
            # Try Streamlit secrets first
            return st.secrets["sqlalchemy_url"]
        except:
            # Fallback to environment variable
            return os.getenv("DATABASE_URL")
    
    def fetch_api_metrics(self) -> Dict:
        """Fetch metrics from the API service."""
        try:
            response = requests.get(f"{self.api_base_url}/metrics", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch API metrics: {e}")
            return self._get_mock_metrics()
    
    def fetch_model_info(self) -> Dict:
        """Fetch model information from API."""
        try:
            response = requests.get(f"{self.api_base_url}/model/info", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch model info: {e}")
            return {
                "model_version": "unknown",
                "threshold": 0.1,
                "feature_names": ["temperature", "humidity"],
                "window_size": 30
            }
    
    def fetch_health_status(self) -> Dict:
        """Fetch service health status."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch health status: {e}")
            return {
                "status": "unknown",
                "model_loaded": False,
                "uptime_seconds": 0,
                "memory_usage_mb": 0
            }
    
    def fetch_recent_anomalies(self, limit: int = 100) -> pd.DataFrame:
        """Fetch recent anomalies from database."""
        if not self.engine:
            return self._generate_mock_anomalies(limit)
        
        try:
            query = text("""
                SELECT TOP (:limit) 
                    id, event_ts, device_id, score, threshold, is_anomaly,
                    feature_json, contrib_json, model_version, inserted_at
                FROM anomalies 
                ORDER BY event_ts DESC
            """)
            
            df = pd.read_sql(query, self.engine, params={"limit": limit})
            df['event_ts'] = pd.to_datetime(df['event_ts'])
            return df
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return self._generate_mock_anomalies(limit)
    
    def _get_mock_metrics(self) -> Dict:
        """Generate mock metrics for development."""
        return {
            "total_requests": np.random.randint(1000, 5000),
            "total_anomalies": np.random.randint(50, 200),
            "avg_processing_time_ms": np.random.uniform(10, 50),
            "p50_processing_time_ms": np.random.uniform(15, 25),
            "p95_processing_time_ms": np.random.uniform(40, 80),
            "p99_processing_time_ms": np.random.uniform(80, 150),
            "anomaly_rate": np.random.uniform(0.02, 0.08),
            "requests_per_second": np.random.uniform(10, 100)
        }
    
    def _generate_mock_anomalies(self, limit: int) -> pd.DataFrame:
        """Generate mock anomaly data for development."""
        now = datetime.now()
        
        data = []
        for i in range(limit):
            ts = now - timedelta(minutes=i * 2)
            device_id = f"dev-{np.random.randint(1, 4)}"
            
            # Generate realistic data
            if np.random.random() < 0.1:  # 10% anomalies
                score = np.random.uniform(0.15, 0.8)
                is_anomaly = True
                temp = np.random.uniform(35, 45)  # High temp anomaly
                humidity = np.random.uniform(20, 80)
            else:
                score = np.random.uniform(0.001, 0.08)
                is_anomaly = False
                temp = np.random.uniform(18, 26)  # Normal temp
                humidity = np.random.uniform(40, 60)
            
            data.append({
                'id': i,
                'event_ts': ts,
                'device_id': device_id,
                'score': score,
                'threshold': 0.1,
                'is_anomaly': is_anomaly,
                'feature_json': json.dumps({'temperature': temp, 'humidity': humidity}),
                'contrib_json': json.dumps({'temperature': score * 0.7, 'humidity': score * 0.3}),
                'model_version': 'ae_v1',
                'inserted_at': ts
            })
        
        return pd.DataFrame(data)
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header">🌡️ IoT Anomaly Detection Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Status indicator
        health = self.fetch_health_status()
        status = health.get('status', 'unknown')
        
        if status == 'healthy':
            st.markdown('<div class="normal-status">✅ System Status: Healthy</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="anomaly-alert">⚠️ System Status: Unhealthy</div>', 
                       unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("📊 Dashboard Controls")
        
        # Refresh settings
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval", 
            [5, 10, 30, 60], 
            index=1,
            help="Dashboard auto-refresh interval in seconds"
        )
        
        # Data filters
        st.sidebar.header("🔍 Data Filters")
        
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"],
            index=1
        )
        
        device_filter = st.sidebar.multiselect(
            "Devices",
            ["dev-1", "dev-2", "dev-3"],
            default=["dev-1", "dev-2", "dev-3"]
        )
        
        # Model info
        st.sidebar.header("🤖 Model Information")
        model_info = self.fetch_model_info()
        
        st.sidebar.metric("Model Version", model_info.get('model_version', 'unknown'))
        st.sidebar.metric("Threshold", f"{model_info.get('threshold', 0.1):.4f}")
        st.sidebar.metric("Window Size", model_info.get('window_size', 30))
        
        return refresh_interval, time_range, device_filter
    
    def render_kpi_metrics(self):
        """Render key performance indicators."""
        st.header("📈 Key Performance Indicators")
        
        # Fetch metrics
        metrics = self.fetch_api_metrics()
        health = self.fetch_health_status()
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔍 Total Requests",
                value=f"{metrics.get('total_requests', 0):,}",
                delta=f"{metrics.get('requests_per_second', 0):.1f} req/s"
            )
        
        with col2:
            anomaly_rate = metrics.get('anomaly_rate', 0)
            st.metric(
                label="🚨 Anomaly Rate",
                value=f"{anomaly_rate:.1%}",
                delta=f"{metrics.get('total_anomalies', 0)} total",
                delta_color="inverse"
            )
        
        with col3:
            avg_time = metrics.get('avg_processing_time_ms', 0)
            st.metric(
                label="⚡ Avg Latency",
                value=f"{avg_time:.1f}ms",
                delta=f"P95: {metrics.get('p95_processing_time_ms', 0):.1f}ms"
            )
        
        with col4:
            uptime_hours = health.get('uptime_seconds', 0) / 3600
            st.metric(
                label="🕐 Uptime",
                value=f"{uptime_hours:.1f}h",
                delta=f"{health.get('memory_usage_mb', 0):.0f}MB RAM"
            )
    
    def render_real_time_chart(self, df: pd.DataFrame):
        """Render real-time anomaly detection chart."""
        st.header("📊 Real-Time Anomaly Detection")
        
        if df.empty:
            st.warning("No data available")
            return
        
        # Parse feature data
        df['temperature'] = df['feature_json'].apply(
            lambda x: json.loads(x).get('temperature', 0) if x else 0
        )
        df['humidity'] = df['feature_json'].apply(
            lambda x: json.loads(x).get('humidity', 0) if x else 0
        )
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Temperature (°C)', 'Humidity (%)', 'Anomaly Score'],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Color anomalies differently
        colors = ['red' if anomaly else 'blue' for anomaly in df['is_anomaly']]
        
        # Temperature plot
        fig.add_trace(
            go.Scatter(
                x=df['event_ts'],
                y=df['temperature'],
                mode='markers+lines',
                name='Temperature',
                marker=dict(color=colors, size=6),
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Humidity plot
        fig.add_trace(
            go.Scatter(
                x=df['event_ts'],
                y=df['humidity'],
                mode='markers+lines',
                name='Humidity',
                marker=dict(color=colors, size=6),
                line=dict(color='green', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Anomaly score plot
        fig.add_trace(
            go.Scatter(
                x=df['event_ts'],
                y=df['score'],
                mode='markers+lines',
                name='Anomaly Score',
                marker=dict(color=colors, size=8),
                line=dict(color='orange', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add threshold line
        threshold = df['threshold'].iloc[0] if not df.empty else 0.1
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: {threshold:.3f}",
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Real-Time Sensor Data and Anomaly Detection",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_table(self, df: pd.DataFrame):
        """Render recent anomalies table."""
        st.header("🚨 Recent Anomalies")
        
        # Filter only anomalies
        anomalies = df[df['is_anomaly'] == True].copy()
        
        if anomalies.empty:
            st.success("No recent anomalies detected! 🎉")
            return
        
        # Prepare display data
        display_df = anomalies.copy()
        display_df['Temperature'] = display_df['feature_json'].apply(
            lambda x: f"{json.loads(x).get('temperature', 0):.1f}°C" if x else "N/A"
        )
        display_df['Humidity'] = display_df['feature_json'].apply(
            lambda x: f"{json.loads(x).get('humidity', 0):.1f}%" if x else "N/A"
        )
        display_df['Score'] = display_df['score'].apply(lambda x: f"{x:.4f}")
        display_df['Time'] = display_df['event_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Select columns for display
        columns_to_show = ['Time', 'device_id', 'Temperature', 'Humidity', 'Score', 'model_version']
        display_columns = {
            'Time': 'Time',
            'device_id': 'Device',
            'Temperature': 'Temperature',
            'Humidity': 'Humidity',
            'Score': 'Anomaly Score',
            'model_version': 'Model'
        }
        
        st.dataframe(
            display_df[columns_to_show].rename(columns=display_columns),
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Anomalies CSV",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def render_device_overview(self, df: pd.DataFrame):
        """Render device-wise overview."""
        st.header("🏭 Device Overview")
        
        if df.empty:
            st.warning("No device data available")
            return
        
        # Group by device
        device_stats = df.groupby('device_id').agg({
            'score': ['count', 'mean', 'max'],
            'is_anomaly': 'sum'
        }).round(4)
        
        device_stats.columns = ['Total_Events', 'Avg_Score', 'Max_Score', 'Anomalies']
        device_stats['Anomaly_Rate'] = (device_stats['Anomalies'] / device_stats['Total_Events'] * 100).round(2)
        
        # Display metrics in columns
        devices = device_stats.index.tolist()
        cols = st.columns(len(devices))
        
        for i, device in enumerate(devices):
            with cols[i]:
                stats = device_stats.loc[device]
                
                st.metric(
                    label=f"📱 {device}",
                    value=f"{stats['Anomaly_Rate']:.1f}%",
                    delta=f"{int(stats['Anomalies'])} anomalies"
                )
                
                st.caption(f"Events: {int(stats['Total_Events'])}")
                st.caption(f"Avg Score: {stats['Avg_Score']:.4f}")
    
    def render_performance_metrics(self):
        """Render detailed performance metrics."""
        st.header("⚡ Performance Metrics")
        
        metrics = self.fetch_api_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Time Distribution")
            
            # Create latency distribution chart
            latency_data = {
                'Percentile': ['P50', 'P95', 'P99'],
                'Latency (ms)': [
                    metrics.get('p50_processing_time_ms', 0),
                    metrics.get('p95_processing_time_ms', 0),
                    metrics.get('p99_processing_time_ms', 0)
                ]
            }
            
            fig = px.bar(
                latency_data, 
                x='Percentile', 
                y='Latency (ms)',
                title="Response Time Percentiles",
                color='Latency (ms)',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("System Health")
            
            health = self.fetch_health_status()
            
            # Health indicators
            indicators = [
                ("Service Status", "✅ Healthy" if health.get('model_loaded') else "❌ Unhealthy"),
                ("Model Loaded", "✅ Yes" if health.get('model_loaded') else "❌ No"),
                ("Memory Usage", f"{health.get('memory_usage_mb', 0):.0f} MB"),
                ("Uptime", f"{health.get('uptime_seconds', 0)/3600:.1f} hours")
            ]
            
            for label, value in indicators:
                st.metric(label, value)
    
    def run(self):
        """Main dashboard execution."""
        # Render components
        self.render_header()
        refresh_interval, time_range, device_filter = self.render_sidebar()
        
        # Main content
        self.render_kpi_metrics()
        
        # Fetch data
        df = self.fetch_recent_anomalies(100)
        
        # Filter by devices if specified
        if device_filter:
            df = df[df['device_id'].isin(device_filter)]
        
        # Render charts and tables
        self.render_real_time_chart(df)
        self.render_device_overview(df)
        self.render_anomaly_table(df)
        self.render_performance_metrics()
        
        # Auto-refresh
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Auto-refresh timer
        time.sleep(refresh_interval)
        st.rerun()


def main():
    """Main function to run the dashboard."""
    dashboard = AnomalyDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
