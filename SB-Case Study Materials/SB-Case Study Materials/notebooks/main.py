import os
import logging
from datetime import datetime
import pandas as pd

# Import from other modules
from preprocess import preprocess_data
from feature_selection import load_data, combine_selected_features
from base_models import load_and_prepare_data, train_and_save_models
from meta_learner import MetaLearner
from evaluation import evaluate_model
from config import (
    DATA_FILES, LOGGING_CONFIG, DATA_DIR, MODELS_DIR, LOGS_DIR,
    EVALUATION, RANDOM_SEED
)

# Set up logging
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'pipeline.log')),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Create necessary directories if they don't exist."""
    # Note: Main directory creation is now handled in config.py
    pass

def run_preprocessing_pipeline():
    """Run the preprocessing pipeline for both rent and sale data."""
    logging.info("Starting preprocessing pipeline...")
    
    # Process both rent and sale data using config parameters
    for data_type, params in DATA_FILES.items():
        preprocess_data(
            params['input'],
            params['output'],
            params['columns_to_remove'],
            data_type
        )

def train_models(model_type):
    """Train base models and meta-learner for specified model type."""
    logging.info(f"Starting model training pipeline for {model_type}...")
    
    # Use paths from config
    data_path = DATA_FILES[model_type]['output']
    target = DATA_FILES[model_type]['target']
    
    # Load and prepare data
    logging.info("Loading and preparing data...")
    data = load_data(data_path)
    selected_features = combine_selected_features(data, target)
    X, y = load_and_prepare_data(data_path, target, list(selected_features))
    
    # Train base models
    logging.info("Training base models...")
    base_models = train_and_save_models(X, y, model_type, output_dir=MODELS_DIR)
    
    # Train meta-learner
    logging.info("Training meta-learner...")
    meta_learner = MetaLearner(MODELS_DIR, model_type)
    meta_learner.fit(X, y)
    meta_learner.save()
    
    return X, y, meta_learner

def evaluate_models(X, y, meta_learner, model_type):
    """Evaluate the performance of the meta-learner."""
    logging.info(f"Evaluating {model_type} models...")
    
    # Make predictions
    predictions = meta_learner.predict(X)
    
    # Evaluate using the evaluation module
    metrics = evaluate_model(y, predictions, model_type)
    
    # Save results to reports directory
    results_file = os.path.join(LOGS_DIR, f'{model_type}_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"{model_type.capitalize()} Model Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Log results
    logging.info(f"{model_type.capitalize()} Model Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    """Main function to run the entire pipeline."""
    start_time = datetime.now()
    logging.info("Starting pipeline execution...")
    
    try:
        # Create necessary directories
        setup_directories()
        
        # Run preprocessing pipeline
        run_preprocessing_pipeline()
        
        # Train and evaluate models for both rent and sale
        results = {}
        for model_type in ['rent', 'sale']:
            # Train models
            X, y, meta_learner = train_models(model_type)
            
            # Evaluate models
            metrics = evaluate_models(X, y, meta_learner, model_type)
            results[model_type] = metrics
        
        # Log final results
        logging.info("\nFinal Results:")
        for model_type, metrics in results.items():
            logging.info(f"\n{model_type.capitalize()} Model:")
            for metric, value in metrics.items():
                logging.info(f"{metric.upper()}: {value:.2f}")
        
        execution_time = datetime.now() - start_time
        logging.info(f"\nTotal execution time: {execution_time}")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise
    
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 