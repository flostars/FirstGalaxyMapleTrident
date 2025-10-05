"""
Real-Time Training Monitor for ExoVision AI
Monitors training progress and provides live updates
"""

import psutil
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TrainingMonitor:
    """Real-time training progress monitor"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.training_log = self.log_dir / "training_progress.json"
        self.metrics_log = self.log_dir / "training_metrics.json"
        
    def find_training_process(self) -> Optional[psutil.Process]:
        """Find the active training process"""
        training_scripts = ['train_real_data.py', 'train.py', 'train_advanced.py', 'quick_train_advanced.py']
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'create_time']):
            try:
                # Check for Python processes
                if proc.info['name'] in ['python.exe', 'python', 'python3.11', 'python3'] and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    # Look for any training script
                    for script in training_scripts:
                        if script in cmdline:
                            return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        proc = self.find_training_process()
        
        if not proc:
            return {
                'status': 'not_running',
                'message': 'No training process found',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Get process info
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()
            create_time = datetime.fromtimestamp(proc.create_time())
            elapsed_time = datetime.now() - create_time
            
            # Estimate progress based on elapsed time
            estimated_total_time = timedelta(hours=2)  # Conservative estimate
            progress_percent = min(100, (elapsed_time.total_seconds() / estimated_total_time.total_seconds()) * 100)
            
            return {
                'status': 'running',
                'pid': proc.pid,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'start_time': create_time.isoformat(),
                'elapsed_time': str(elapsed_time).split('.')[0],
                'progress_percent': progress_percent,
                'estimated_completion': (create_time + estimated_total_time).isoformat(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error monitoring process: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def log_training_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics to file"""
        try:
            # Load existing metrics
            if self.metrics_log.exists():
                with open(self.metrics_log, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            # Add timestamp and new metrics
            metrics['timestamp'] = datetime.now().isoformat()
            all_metrics.append(metrics)
            
            # Keep only last 1000 entries
            if len(all_metrics) > 1000:
                all_metrics = all_metrics[-1000:]
            
            # Save updated metrics
            with open(self.metrics_log, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            print(f"Error logging metrics: {e}")
    
    def get_training_metrics_history(self) -> List[Dict[str, Any]]:
        """Get historical training metrics"""
        try:
            if self.metrics_log.exists():
                with open(self.metrics_log, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading metrics history: {e}")
            return []
    
    def create_progress_chart(self) -> go.Figure:
        """Create training progress chart"""
        metrics_history = self.get_training_metrics_history()
        
        if not metrics_history:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No training data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="white")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            return fig
        
        # Extract data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics_history]
        losses = [m.get('loss', 0) for m in metrics_history]
        accuracies = [m.get('accuracy', 0) for m in metrics_history]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Loss', 'Validation Accuracy'),
            vertical_spacing=0.1
        )
        
        # Loss plot
        fig.add_trace(go.Scatter(
            x=timestamps, y=losses,
            mode='lines+markers',
            name='Loss',
            line=dict(color='#f472b6', width=2),
            marker=dict(size=4)
        ), row=1, col=1)
        
        # Accuracy plot
        fig.add_trace(go.Scatter(
            x=timestamps, y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#00f5ff', width=2),
            marker=dict(size=4)
        ), row=2, col=1)
        
        fig.update_layout(
            title="Training Progress",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_system_monitor_chart(self) -> go.Figure:
        """Create system resource monitoring chart"""
        metrics_history = self.get_training_metrics_history()
        
        if not metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No system data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="white")
            )
            return fig
        
        # Extract system data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics_history]
        cpu_usage = [m.get('cpu_percent', 0) for m in metrics_history]
        memory_usage = [m.get('memory_mb', 0) for m in metrics_history]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (MB)'),
            vertical_spacing=0.1
        )
        
        # CPU usage
        fig.add_trace(go.Scatter(
            x=timestamps, y=cpu_usage,
            mode='lines+markers',
            name='CPU %',
            line=dict(color='#8b5cf6', width=2),
            fill='tonexty'
        ), row=1, col=1)
        
        # Memory usage
        fig.add_trace(go.Scatter(
            x=timestamps, y=memory_usage,
            mode='lines+markers',
            name='Memory (MB)',
            line=dict(color='#10b981', width=2),
            fill='tonexty'
        ), row=2, col=1)
        
        fig.update_layout(
            title="System Resource Usage",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            showlegend=False
        )
        
        return fig
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        status = self.get_training_status()
        metrics_history = self.get_training_metrics_history()
        
        summary = {
            'current_status': status,
            'total_metrics_points': len(metrics_history),
            'last_update': datetime.now().isoformat()
        }
        
        if metrics_history:
            # Calculate statistics
            losses = [m.get('loss', 0) for m in metrics_history if 'loss' in m]
            accuracies = [m.get('accuracy', 0) for m in metrics_history if 'accuracy' in m]
            
            if losses:
                summary['loss_stats'] = {
                    'min': min(losses),
                    'max': max(losses),
                    'latest': losses[-1],
                    'trend': 'decreasing' if len(losses) > 1 and losses[-1] < losses[0] else 'stable'
                }
            
            if accuracies:
                summary['accuracy_stats'] = {
                    'min': min(accuracies),
                    'max': max(accuracies),
                    'latest': accuracies[-1],
                    'trend': 'increasing' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'stable'
                }
        
        return summary
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        print("🔍 Starting Training Monitor")
        print("=" * 50)
        
        try:
            while True:
                # Get current status
                status = self.get_training_status()
                
                if status['status'] == 'running':
                    # Log current metrics
                    self.log_training_metrics({
                        'cpu_percent': status.get('cpu_percent', 0),
                        'memory_mb': status.get('memory_mb', 0),
                        'progress_percent': status.get('progress_percent', 0)
                    })
                    
                    print(f"📊 Training Progress: {status.get('progress_percent', 0):.1f}% | "
                          f"CPU: {status.get('cpu_percent', 0):.1f}% | "
                          f"Memory: {status.get('memory_mb', 0):.1f}MB")
                else:
                    print(f"❌ Training Status: {status['message']}")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")


class StreamlitTrainingMonitor:
    """Streamlit integration for training monitoring"""
    
    def __init__(self):
        self.monitor = TrainingMonitor()
    
    def display_training_status(self):
        """Display training status in Streamlit"""
        import streamlit as st
        
        status = self.monitor.get_training_status()
        
        if status['status'] == 'running':
            # Progress bar
            progress = status.get('progress_percent', 0) / 100
            st.progress(progress)
            
            # Status metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Progress", f"{status.get('progress_percent', 0):.1f}%")
            
            with col2:
                st.metric("CPU Usage", f"{status.get('cpu_percent', 0):.1f}%")
            
            with col3:
                st.metric("Memory", f"{status.get('memory_mb', 0):.1f}MB")
            
            with col4:
                st.metric("Elapsed", status.get('elapsed_time', 'Unknown'))
            
            # Progress charts
            st.subheader("📈 Training Progress")
            progress_chart = self.monitor.create_progress_chart()
            st.plotly_chart(progress_chart, use_container_width=True)
            
            # System monitoring
            st.subheader("💻 System Resources")
            system_chart = self.monitor.create_system_monitor_chart()
            st.plotly_chart(system_chart, use_container_width=True)
            
        else:
            st.error(f"Training Status: {status['message']}")
            
            # Show helpful information
            st.info("""
            **💡 To start training monitoring:**
            1. Go to the **Neural Training** tab
            2. Upload training data or use default dataset
            3. Click **Train/Retrain** to start training
            4. Return to this tab to monitor progress
            """)
    
    def display_training_summary(self):
        """Display training summary"""
        import streamlit as st
        
        summary = self.monitor.get_training_summary()
        
        st.subheader("📊 Training Summary")
        
        # Current status
        status = summary['current_status']
        if status['status'] == 'running':
            st.success("✅ Training is running")
        else:
            st.warning(f"⚠️ {status['message']}")
            
            # Show system information when no training is running
            st.subheader("🖥️ System Information")
            
            try:
                import psutil
                
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("CPU Usage", f"{cpu_percent:.1f}%")
                
                with col2:
                    st.metric("Memory Usage", f"{memory.percent:.1f}%")
                
                with col3:
                    st.metric("Disk Usage", f"{disk.percent:.1f}%")
                    
            except Exception as e:
                st.error(f"Error getting system info: {e}")
        
        # Statistics
        if 'loss_stats' in summary:
            st.subheader("📉 Loss Statistics")
            loss_stats = summary['loss_stats']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Loss", f"{loss_stats['latest']:.4f}")
            with col2:
                st.metric("Min Loss", f"{loss_stats['min']:.4f}")
            with col3:
                st.metric("Trend", loss_stats['trend'])
        
        if 'accuracy_stats' in summary:
            st.subheader("🎯 Accuracy Statistics")
            acc_stats = summary['accuracy_stats']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Accuracy", f"{acc_stats['latest']:.4f}")
            with col2:
                st.metric("Max Accuracy", f"{acc_stats['max']:.4f}")
            with col3:
                st.metric("Trend", acc_stats['trend'])


def test_monitoring():
    """Test the monitoring system"""
    print("🔍 Testing Training Monitor")
    print("=" * 50)
    
    monitor = TrainingMonitor()
    
    # Test status check
    status = monitor.get_training_status()
    print(f"Training Status: {status}")
    
    # Test metrics logging
    test_metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'cpu_percent': 65.2,
        'memory_mb': 1024
    }
    monitor.log_training_metrics(test_metrics)
    print("✅ Metrics logged successfully")
    
    # Test summary
    summary = monitor.get_training_summary()
    print(f"Training Summary: {summary}")
    
    print("\n🎉 Monitoring system test completed!")


if __name__ == "__main__":
    test_monitoring()
