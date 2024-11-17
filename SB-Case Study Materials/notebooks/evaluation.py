import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import EVALUATION, PLOTS_DIR, REPORTS_DIR, LOGS_DIR

class ModelEvaluator:
    """Class to handle model evaluation metrics and visualization."""
    
    def __init__(self, model_type):
        """
        Initialize ModelEvaluator.
        
        Args:
            model_type (str): Type of model ('rent' or 'sale')
        """
        self.model_type = model_type
        self.metrics = {}
        
        # Set up logging using config parameters
        self.logger = logging.getLogger(f'{model_type}_evaluator')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            os.makedirs(LOGS_DIR, exist_ok=True)
            handler = logging.FileHandler(os.path.join(LOGS_DIR, f'{model_type}_evaluation.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary containing calculated metrics
        """
        try:
            # Calculate basic metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
            
            self.metrics = {
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae,
                'MSE': mse,
                'MAPE': mape,
                'Explained_Variance': explained_variance
            }
            
            self.logger.info(f"Metrics calculated successfully for {self.model_type} model")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_residuals(self, y_true, y_pred, save_dir=PLOTS_DIR):
        """
        Create and save residual plots.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_dir: Directory to save plots
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            residuals = y_true - y_pred
            
            # Use config figure size
            plt.figure(figsize=EVALUATION['plot_figsize'])
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot - {self.model_type.capitalize()} Model')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.savefig(os.path.join(save_dir, f'{self.model_type}_residuals.png'))
            plt.close()
            
            # Create QQ plot
            plt.figure(figsize=EVALUATION['plot_figsize'])
            sns.histplot(residuals, kde=True)
            plt.title(f'Residuals Distribution - {self.model_type.capitalize()} Model')
            plt.xlabel('Residuals')
            plt.savefig(os.path.join(save_dir, f'{self.model_type}_residuals_dist.png'))
            plt.close()
            
            self.logger.info(f"Residual plots saved in {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating residual plots: {str(e)}")
            raise

    def plot_actual_vs_predicted(self, y_true, y_pred, save_dir=PLOTS_DIR):
        """
        Create and save actual vs predicted scatter plot.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_dir: Directory to save plots
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            plt.figure(figsize=EVALUATION['plot_figsize'])
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted - {self.model_type.capitalize()} Model')
            plt.savefig(os.path.join(save_dir, f'{self.model_type}_actual_vs_predicted.png'))
            plt.close()
            
            self.logger.info(f"Actual vs Predicted plot saved in {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating actual vs predicted plot: {str(e)}")
            raise

    def generate_evaluation_report(self, y_true, y_pred, output_dir=REPORTS_DIR):
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            output_dir: Directory to save the report
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred)
            
            # Create plots using config directories
            self.plot_residuals(y_true, y_pred, PLOTS_DIR)
            self.plot_actual_vs_predicted(y_true, y_pred, PLOTS_DIR)
            
            # Generate report with configured metrics
            report = f"""
Model Evaluation Report - {self.model_type.capitalize()} Model
================================================

Metrics Summary:
---------------
"""
            # Only include metrics specified in config
            for metric in EVALUATION['metrics']:
                if metric in metrics:
                    report += f"{metric}: {metrics[metric]:.4f}\n"
            
            # Generate report
            report = f"""
Model Evaluation Report - {self.model_type.capitalize()} Model
================================================

Metrics Summary:
---------------
RMSE: {metrics['RMSE']:.2f}
R² Score: {metrics['R2']:.4f}
MAE: {metrics['MAE']:.2f}
MSE: {metrics['MSE']:.2f}
MAPE: {metrics['MAPE']:.2f}%
Explained Variance: {metrics['Explained_Variance']:.4f}

Additional Statistics:
--------------------
Number of samples: {len(y_true)}
Mean of actual values: {np.mean(y_true):.2f}
Mean of predicted values: {np.mean(y_pred):.2f}
Standard deviation of residuals: {np.std(y_true - y_pred):.2f}

Plots generated:
--------------
- Residual plot: plots/{self.model_type}_residuals.png
- Residuals distribution: plots/{self.model_type}_residuals_dist.png
- Actual vs Predicted: plots/{self.model_type}_actual_vs_predicted.png
"""
            
            # Save report
            report_path = os.path.join(output_dir, f'{self.model_type}_evaluation_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Evaluation report generated and saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            raise

def evaluate_model(y_true, y_pred, model_type):
    """
    Convenience function to evaluate a model and generate full report.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_type: Type of model ('rent' or 'sale')
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    evaluator = ModelEvaluator(model_type)
    evaluator.generate_evaluation_report(y_true, y_pred)
    return evaluator.metrics
