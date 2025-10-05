"""
Experiment Tracking for FSRPPO

 experiment tracking and logging system for FSRPPO training.
Provides detailed logging, metrics tracking, and result visualization.

Features:
- Real-time metrics logging
- Experiment configuration tracking
- Result visualization and analysis
- Model performance comparison
- Export capabilities for further analysis
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExperimentTracker:
    """
     experiment tracking system
    
    This class provides detailed tracking of FSRPPO experiments including
    metrics, configurations, and results for analysis and comparison.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    save_dir : str or Path
        Directory to save experiment data
    auto_save : bool, default=True
        Whether to automatically save metrics
    save_frequency : int, default=100
        Frequency of automatic saves (in log calls)
    """
    
    def __init__(
        self,
        experiment_name: str,
        save_dir: Union[str, Path] = './experiments',
        auto_save: bool = True,
        save_frequency: int = 100
    ):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.auto_save = auto_save
        self.save_frequency = save_frequency
        
        # Create experiment directory
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking data
        self.metrics = []
        self.config = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_duration': None,
            'status': 'running'
        }
        
        # Counters
        self.log_count = 0
        self.start_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_experiment_logging()
        
        self.logger.info(f"Experiment tracker initialized: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_experiment_logging(self):
        """Setup experiment-specific logging"""
        
        # Create log file handler
        log_file = self.experiment_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set experiment configuration
        
        Parameters:
        -----------
        config : dict
            Experiment configuration dictionary
        """
        self.config = config.copy()
        
        # Save config immediately
        config_file = self.experiment_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        self.logger.info("Experiment configuration saved")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int, str]]):
        """
        Log metrics for current step
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metrics to log
        """
        # Add timestamp
        timestamped_metrics = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }
        
        # Store metrics
        self.metrics.append(timestamped_metrics)
        self.log_count += 1
        
        # Auto-save if enabled
        if self.auto_save and self.log_count % self.save_frequency == 0:
            self._save_metrics()
        
        # Log key metrics
        if 'episode' in metrics:
            episode = metrics['episode']
            if episode % 100 == 0:  # Log every 100 episodes
                key_metrics = {k: v for k, v in metrics.items() 
                             if k in ['episode_reward', 'total_return', 'sharpe_ratio']}
                self.logger.info(f"Episode {episode}: {key_metrics}")
    
    def log_milestone(self, milestone: str, data: Optional[Dict] = None):
        """
        Log important milestones
        
        Parameters:
        -----------
        milestone : str
            Description of the milestone
        data : dict, optional
            Additional data associated with the milestone
        """
        milestone_entry = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            'milestone': milestone,
            'data': data or {}
        }
        
        # Save milestone
        milestones_file = self.experiment_dir / 'milestones.json'
        
        if milestones_file.exists():
            with open(milestones_file, 'r') as f:
                milestones = json.load(f)
        else:
            milestones = []
        
        milestones.append(milestone_entry)
        
        with open(milestones_file, 'w') as f:
            json.dump(milestones, f, indent=2, default=str)
        
        self.logger.info(f"Milestone: {milestone}")
    
    def save_results(self, results: Dict[str, Any]):
        """
        Save final experiment results
        
        Parameters:
        -----------
        results : dict
            Final experiment results
        """
        # Update metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_duration'] = time.time() - self.start_time
        self.metadata['status'] = 'completed'
        self.metadata['total_metrics_logged'] = len(self.metrics)
        
        # Save results
        results_file = self.experiment_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metadata
        metadata_file = self.experiment_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Final save of metrics
        self._save_metrics()
        
        self.logger.info("Experiment results saved")
        self.logger.info(f"Total duration: {self.metadata['total_duration']:.2f} seconds")
    
    def _save_metrics(self):
        """Save metrics to file"""
        
        # Save as JSON
        metrics_file = self.experiment_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            csv_file = self.experiment_dir / 'metrics.csv'
            df.to_csv(csv_file, index=False)
        
        # Save as pickle for Python analysis
        pickle_file = self.experiment_dir / 'metrics.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics as pandas DataFrame
        
        Returns:
        --------
        pd.DataFrame
            Metrics DataFrame
        """
        if not self.metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics)
    
    def plot_metrics(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (15, 10)
    ):
        """
        Plot experiment metrics
        
        Parameters:
        -----------
        metrics_to_plot : List[str], optional
            List of metrics to plot. If None, plots common metrics
        save_path : str, optional
            Path to save the plot
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if not self.metrics:
            self.logger.warning("No metrics to plot")
            return
        
        df = self.get_metrics_dataframe()
        
        # Default metrics to plot
        if metrics_to_plot is None:
            available_metrics = df.columns.tolist()
            metrics_to_plot = []
            
            # Common trading metrics
            common_metrics = [
                'episode_reward', 'total_return', 'sharpe_ratio', 
                'max_drawdown', 'portfolio_value', 'policy_loss'
            ]
            
            for metric in common_metrics:
                if metric in available_metrics:
                    metrics_to_plot.append(metric)
        
        # Filter available metrics
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
        
        if not metrics_to_plot:
            self.logger.warning("No valid metrics found to plot")
            return
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                ax = axes[i]
                
                # Plot metric over time
                if 'episode' in df.columns:
                    x = df['episode']
                    xlabel = 'Episode'
                elif 'elapsed_time' in df.columns:
                    x = df['elapsed_time'] / 3600  # Convert to hours
                    xlabel = 'Time (hours)'
                else:
                    x = range(len(df))
                    xlabel = 'Step'
                
                y = df[metric]
                
                # Remove NaN values
                mask = ~(pd.isna(x) | pd.isna(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) > 0:
                    ax.plot(x_clean, y_clean, linewidth=1, alpha=0.7)
                    
                    # Add rolling average if enough data
                    if len(y_clean) > 20:
                        window = min(50, len(y_clean) // 10)
                        rolling_avg = pd.Series(y_clean).rolling(window=window, center=True).mean()
                        ax.plot(x_clean, rolling_avg, linewidth=2, color='red', alpha=0.8, label=f'Rolling Avg ({window})')
                        ax.legend()
                
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel(xlabel)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.experiment_dir / 'metrics_plot.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Metrics plot saved: {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate experiment report
        
        Returns:
        --------
        str
            Formatted experiment report
        """
        if not self.metrics:
            return "No metrics available for report generation"
        
        df = self.get_metrics_dataframe()
        
        # Basic statistics
        report = f"""
FSRPPO Experiment Report
========================

Experiment: {self.experiment_name}
Start Time: {self.metadata['start_time']}
End Time: {self.metadata.get('end_time', 'Running')}
Duration: {self.metadata.get('total_duration', time.time() - self.start_time):.2f} seconds
Status: {self.metadata['status']}
Total Metrics Logged: {len(self.metrics)}

"""
        
        # Key metrics summary
        if 'episode' in df.columns:
            total_episodes = df['episode'].max()
            report += f"Total Episodes: {total_episodes}\n"
        
        # Performance metrics
        performance_metrics = ['episode_reward', 'total_return', 'sharpe_ratio', 'max_drawdown']
        
        report += "\nPerformance Summary:\n"
        report += "-" * 20 + "\n"
        
        for metric in performance_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    report += f"{metric.replace('_', ' ').title()}:\n"
                    report += f"  Mean: {values.mean():.4f}\n"
                    report += f"  Std:  {values.std():.4f}\n"
                    report += f"  Min:  {values.min():.4f}\n"
                    report += f"  Max:  {values.max():.4f}\n"
                    
                    # Final value
                    if len(values) > 0:
                        report += f"  Final: {values.iloc[-1]:.4f}\n"
                    report += "\n"
        
        # Configuration summary
        if self.config:
            report += "Configuration:\n"
            report += "-" * 13 + "\n"
            
            # Key configuration items
            key_configs = ['agent', 'environment', 'fsr', 'preprocessing']
            
            for key in key_configs:
                if key in self.config:
                    report += f"{key.title()}:\n"
                    config_section = self.config[key]
                    if isinstance(config_section, dict):
                        for sub_key, value in config_section.items():
                            report += f"  {sub_key}: {value}\n"
                    else:
                        report += f"  {config_section}\n"
                    report += "\n"
        
        return report
    
    def save_report(self, filename: Optional[str] = None):
        """
        Save experiment report to file
        
        Parameters:
        -----------
        filename : str, optional
            Filename for the report. If None, uses default name
        """
        if filename is None:
            filename = 'experiment_report.txt'
        
        report = self.generate_report()
        
        report_path = self.experiment_dir / filename
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Experiment report saved: {report_path}")
    
    def compare_with_experiment(self, other_experiment_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Compare current experiment with another experiment
        
        Parameters:
        -----------
        other_experiment_dir : str or Path
            Directory of the other experiment
            
        Returns:
        --------
        dict
            Comparison results
        """
        other_dir = Path(other_experiment_dir)
        
        # Load other experiment's metrics
        other_metrics_file = other_dir / 'metrics.json'
        if not other_metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {other_metrics_file}")
        
        with open(other_metrics_file, 'r') as f:
            other_metrics = json.load(f)
        
        # Convert to DataFrames
        current_df = self.get_metrics_dataframe()
        other_df = pd.DataFrame(other_metrics)
        
        # Compare key metrics
        comparison = {}
        
        key_metrics = ['episode_reward', 'total_return', 'sharpe_ratio', 'max_drawdown']
        
        for metric in key_metrics:
            if metric in current_df.columns and metric in other_df.columns:
                current_values = current_df[metric].dropna()
                other_values = other_df[metric].dropna()
                
                if len(current_values) > 0 and len(other_values) > 0:
                    comparison[metric] = {
                        'current_mean': current_values.mean(),
                        'other_mean': other_values.mean(),
                        'current_final': current_values.iloc[-1],
                        'other_final': other_values.iloc[-1],
                        'improvement_mean': current_values.mean() - other_values.mean(),
                        'improvement_final': current_values.iloc[-1] - other_values.iloc[-1]
                    }
        
        return comparison


# Example usage and testing
if __name__ == "__main__":
    import logging
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Experiment Tracker")
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_name='test_experiment',
        save_dir='./test_experiments',
        auto_save=True,
        save_frequency=5
    )
    
    # Set configuration
    config = {
        'agent': {'lr': 1e-5, 'gamma': 0.99},
        'environment': {'initial_cash': 10000},
        'fsr': {'hurst_threshold': 0.5}
    }
    tracker.set_config(config)
    
    # Simulate training with metrics logging
    print("Simulating training...")
    
    for episode in range(20):
        # Simulate metrics
        metrics = {
            'episode': episode,
            'episode_reward': np.random.normal(0.1, 0.5),
            'total_return': np.random.normal(0.05, 0.2),
            'sharpe_ratio': np.random.normal(1.0, 0.5),
            'max_drawdown': np.random.uniform(-0.3, -0.05),
            'portfolio_value': 10000 * (1 + np.random.normal(0.05, 0.1))
        }
        
        tracker.log_metrics(metrics)
        
        # Log milestone
        if episode == 10:
            tracker.log_milestone("Halfway point", {'episode': episode})
        
        time.sleep(0.1)  # Simulate time passing
    
    # Save final results
    results = {
        'final_performance': {
            'mean_return': 0.08,
            'mean_sharpe': 1.2,
            'best_episode': 15
        },
        'training_completed': True
    }
    
    tracker.save_results(results)
    
    # Generate and save report
    print("\nGenerating report...")
    report = tracker.generate_report()
    print(report)
    
    tracker.save_report()
    
    # Plot metrics
    print("Plotting metrics...")
    tracker.plot_metrics()
    
    print("\nExperiment Tracker test completed successfully!")
    print(f"Experiment data saved in: {tracker.experiment_dir}")