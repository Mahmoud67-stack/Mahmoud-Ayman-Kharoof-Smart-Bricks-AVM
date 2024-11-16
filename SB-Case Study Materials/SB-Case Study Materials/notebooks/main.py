import os
import logging
from datetime import datetime
import pandas as pd

# Import from other modules
from notebooks.preprocess import preprocess_data
from notebooks.feature_selection import load_data, combine_selected_features
from notebooks.base_models import load_and_prepare_data, train_and_save_models
from notebooks.meta_learner import MetaLearner
from notebooks.evaluation import evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_preprocessing_pipeline():
    """Run the preprocessing pipeline for both rent and sale data."""
    logging.info("Starting preprocessing pipeline...")
    
    # Columns to remove for rentals
    rentals_columns_to_remove = [
        'total_properties', 'contract_amount', 'registration_date', 'ejari_contract_number',
        'version_text', 'is_freehold_text', 'property_type_ar', 'property_subtype_ar',
        'property_usage_ar', 'property_usage_id', 'project_name_ar', 'area_ar', 'area_id',
        'parcel_id', 'property_id', 'land_property_id', 'nearest_landmark_ar',
        'nearest_metro_ar', 'nearest_mall_ar', 'master_project_ar',
        'ejari_property_type_id', 'ejari_property_sub_type_id', 'entry_id',
        'meta_ts', 'req_from', 'req_to'
    ]

    # Columns to remove for transactions
    transactions_columns_to_remove = [
        'transaction_size_sqm', 'transaction_number', 'transaction_type_id',
        'property_usage_id', 'transaction_subtype_id', 'property_id',
        'property_type_ar', 'property_type_id', 'property_subtype_ar',
        'property_subtype_id', 'building_age', 'rooms_ar', 'project_name_ar',
        'transaction_subtype_id', 'area_ar', 'area_id', 'nearest_landmark_ar',
        'nearest_metro_ar', 'nearest_mall_ar', 'master_project_ar',
        'entry_id', 'meta_ts', 'parcel_id', 'req_from', 'req_to'
    ]

    # Process rent data
    rent_input = os.path.join('data', 'snp_dld_2024_rents.csv')
    rent_output = os.path.join('data', 'snp_dld_2024_rents.csv')
    preprocess_data(rent_input, rent_output, rentals_columns_to_remove, 'rent')

    # Process sale data
    sale_input = os.path.join('data', 'snp_dld_2024_transactions.csv')
    sale_output = os.path.join('data', 'snp_dld_2024_transactions.csv')
    preprocess_data(sale_input, sale_output, transactions_columns_to_remove, 'sale')

def train_models(model_type):
    """Train base models and meta-learner for specified model type."""
    logging.info(f"Starting model training pipeline for {model_type}...")
    
    # Define paths and variables
    data_path = os.path.join(
        'data', 
        f'snp_dld_2024_{"rents" if model_type == "rent" else "transactions"}_cleaned.parquet'
    )
    target = 'annual_amount' if model_type == 'rent' else 'amount'
    
    # Load and prepare data
    logging.info("Loading and preparing data...")
    data = load_data(data_path)
    selected_features = combine_selected_features(data, target)
    X, y = load_and_prepare_data(data_path, target, list(selected_features))
    
    # Train base models
    logging.info("Training base models...")
    base_models = train_and_save_models(X, y, model_type, output_dir='models')
    
    # Train meta-learner
    logging.info("Training meta-learner...")
    meta_learner = MetaLearner('models', model_type)
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