import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from bayes_opt import BayesianOptimization
import joblib
import os
from config import (
    MODEL_PARAMS, 
    MODELS_DIR, 
    DATA_FILES, 
    EVALUATION,
    RANDOM_SEED,
    PLOTS_DIR
)
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_prepare_data(file_path, target_variable, selected_features):
    """
    Load data and prepare X and y for modeling.
    """
    data = pd.read_parquet(file_path)
    # Ensure target variable is not in features
    features = [f for f in selected_features if f != target_variable]
    X = data[features]
    y = data[target_variable]
    
    return X, y

def plot_optimization_results(optimizer, model_name, model_type):
    """Plot the optimization process results"""
    # Extract optimization history
    targets = np.array([res['target'] for res in optimizer.res])
    iterations = range(len(targets))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, -targets, 'b-', label='Objective value')
    plt.scatter(iterations, -targets, c='b')
    
    plt.title(f'{model_name} Optimization Process - {model_type}')
    plt.xlabel('Iteration')
    plt.ylabel('Negative MSE')
    plt.legend()
    plt.grid(True)
    
    # Save the plot in plots directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'optimization_{model_name}_{model_type}_{timestamp}.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved optimization plot to: {filepath}")

def plot_cv_results(cv_scores, model_name, model_type):
    """Plot cross-validation results"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=cv_scores)
    plt.title(f'{model_name} Cross-Validation Scores - {model_type}')
    plt.ylabel('Negative MSE')
    
    # Save the plot in plots directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'cv_scores_{model_name}_{model_type}_{timestamp}.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved CV scores plot to: {filepath}")

def optimize_xgboost(X, y, model_type):
    """Optimize XGBoost hyperparameters using Bayesian Optimization."""
    def xgb_evaluate(max_depth, learning_rate, n_estimators, min_child_weight, subsample):
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            **MODEL_PARAMS['xgboost']['default_params'],
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Plot cross-validation results
        plot_cv_results(scores, 'XGBoost', model_type)
        
        return scores.mean()

    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=MODEL_PARAMS['xgboost']['pbounds'],
        random_state=RANDOM_SEED
    )
    
    optimizer.maximize(init_points=3, n_iter=5)
    
    # Plot optimization results
    plot_optimization_results(optimizer, 'XGBoost', model_type)
    
    return optimizer.max

def optimize_random_forest(X, y, model_type):
    """Optimize Random Forest hyperparameters using Bayesian Optimization."""
    def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            **MODEL_PARAMS['random_forest']['default_params']
        }
        
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, X, y, cv=EVALUATION['cv_folds'], 
                               scoring='neg_mean_squared_error')
        
        # Plot cross-validation results
        plot_cv_results(scores, 'Random Forest', model_type)
        
        return scores.mean()

    optimizer = BayesianOptimization(
        f=rf_evaluate,
        pbounds=MODEL_PARAMS['random_forest']['pbounds'],
        random_state=RANDOM_SEED
    )
    
    optimizer.maximize(init_points=3, n_iter=5)
    
    # Plot optimization results
    plot_optimization_results(optimizer, 'Random Forest', model_type)
    
    return optimizer.max

def optimize_svr(X, y, model_type):
    """Optimize SVR hyperparameters using Bayesian Optimization."""
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y

    def svr_evaluate(C, epsilon, gamma):
        params = {
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma,
            'kernel': 'rbf',
            'cache_size': 2000
        }
        model = SVR(**params)
        scores = cross_val_score(
            model, 
            X_sample, 
            y_sample, 
            cv=2,
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        # Plot cross-validation results
        plot_cv_results(scores, 'SVR', model_type)
        
        return scores.mean()

    optimizer = BayesianOptimization(
        f=svr_evaluate,
        pbounds={
            'C': (0.1, 100.0),
            'epsilon': (0.01, 0.5),
            'gamma': (0.01, 0.5)
        },
        random_state=RANDOM_SEED
    )
    
    optimizer.maximize(init_points=2, n_iter=2)
    
    # Plot optimization results
    plot_optimization_results(optimizer, 'SVR', model_type)
    
    return optimizer.max

def train_and_save_models(X, y, model_type, output_dir=MODELS_DIR):
    """
    Train models with optimized parameters and save them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def train_model(model_name):
        try:
            logging.info(f"Starting {model_name} training for {model_type} with data shape: X={X.shape}, y={y.shape}")
            if model_name == 'xgboost':
                logging.info(f"Starting XGBoost optimization for {model_type}")
                params = optimize_xgboost(X, y, model_type)
                logging.info(f"XGBoost optimization completed for {model_type}")
                model = xgb.XGBRegressor(
                    max_depth=int(params['params']['max_depth']),
                    learning_rate=params['params']['learning_rate'],
                    n_estimators=int(params['params']['n_estimators']),
                    min_child_weight=params['params']['min_child_weight'],
                    subsample=params['params']['subsample'],
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    n_jobs=-1
                )
            elif model_name == 'random_forest':
                logging.info(f"Starting Random Forest optimization for {model_type}")
                params = optimize_random_forest(X, y, model_type)
                logging.info(f"Random Forest optimization completed for {model_type}")
                model = RandomForestRegressor(
                    n_estimators=int(params['params']['n_estimators']),
                    max_depth=int(params['params']['max_depth']),
                    min_samples_split=int(params['params']['min_samples_split']),
                    min_samples_leaf=int(params['params']['min_samples_leaf']),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'svr':
                logging.info(f"Starting SVR optimization for {model_type}")
                params = optimize_svr(X, y, model_type)
                logging.info(f"SVR optimization completed for {model_type}")
                
                if len(X) > 10000:  
                    idx = np.random.choice(len(X), 10000, replace=False)
                    X_train = X.iloc[idx]
                    y_train = y.iloc[idx]
                    logging.info(f"Using reduced dataset for SVR training: {len(X_train)} samples")
                else:
                    X_train = X
                    y_train = y
                
                model = SVR(
                    C=params['params']['C'],
                    epsilon=params['params']['epsilon'],
                    gamma=params['params']['gamma'],
                    kernel='rbf',
                    cache_size=3000,  # Increased cache
                    max_iter=1000     # Add iteration limit
                )
                
                logging.info(f"Training SVR for {model_type}")
                model.fit(X_train, y_train)
                
                # Save model immediately after training
                model_path = f'{output_dir}/svr_{model_type}.joblib'
                joblib.dump(model, model_path)
                logging.info(f"Saved SVR model to {model_path}")
                
                return model_name, model
            else:  # svr
                logging.info(f"Starting SVR optimization for {model_type}")
                params = optimize_svr(X, y, model_type)
                logging.info(f"SVR optimization completed for {model_type}")
                model = SVR(
                    C=params['params']['C'],
                    epsilon=params['params']['epsilon'],
                    gamma=params['params']['gamma'],
                    kernel='rbf'
                )
            
            logging.info(f"Training {model_name} for {model_type}")
            model.fit(X, y)
            model_path = f'{output_dir}/{model_name}_{model_type}.joblib'
            joblib.dump(model, model_path)
            logging.info(f"Saved {model_name} model to {model_path}")
            return model_name, model
            
        except Exception as e:
            logging.error(f"Error in {model_name} training for {model_type}: {str(e)}", exc_info=True)
            return model_name, None

    try:
        # Reduce parallel jobs to 1 to avoid resource conflicts
        results = Parallel(n_jobs=1, timeout=7200)(
            delayed(train_model)(model_name) 
            for model_name in ['xgboost', 'random_forest', 'svr']
        )
        return {name: model for name, model in results if model is not None}
    except Exception as e:
        logging.error(f"Error in parallel processing for {model_type}: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    start_time = datetime.now()
    logging.info(f"Starting model training at {start_time}")

    try:
        # Import feature selection functions
        from feature_selection import load_data, combine_selected_features
        
        # Get file paths from config
        rent_data_path = DATA_FILES['rent']['output']
        sale_data_path = DATA_FILES['sale']['output']
        
        # Define target variables from config
        rent_target = DATA_FILES['rent']['target']
        sale_target = DATA_FILES['sale']['target']
        
        # Process both datasets in parallel
        def process_dataset(data_type):
            try:
                logging.info(f"Processing {data_type} dataset")
                data_path = rent_data_path if data_type == 'rent' else sale_data_path
                target = rent_target if data_type == 'rent' else sale_target
                
                logging.info(f"Loading data for {data_type} from {data_path}")
                data = load_data(data_path)
                logging.info(f"Data loaded for {data_type}, shape: {data.shape}")
                
                logging.info(f"Selecting features for {data_type}")
                selected_features = combine_selected_features(data, target)
                # Ensure target is not in selected features
                selected_features = [f for f in selected_features if f != target]
                logging.info(f"Selected features for {data_type}: {selected_features}")
                
                X, y = load_and_prepare_data(data_path, target, selected_features)
                logging.info(f"Prepared data for {data_type}, X shape: {X.shape}, y shape: {y.shape}")
                
                return train_and_save_models(X, y, data_type)
            except Exception as e:
                logging.error(f"Error processing {data_type} dataset: {str(e)}", exc_info=True)
                raise e

        # Process datasets in parallel using joblib
        results = Parallel(n_jobs=2)(
            delayed(process_dataset)(data_type) 
            for data_type in ['rent', 'sale']
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"All processing completed. Total duration: {duration}")
        sys.exit(0)  # Explicitly exit
        
    except Exception as e:
        logging.error(f"Fatal error in main process: {str(e)}")
        sys.exit(1)  # Exit with error code
  