import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import (
    MODEL_PARAMS, 
    MODELS_DIR, 
    DATA_FILES, 
    EVALUATION,
    RANDOM_SEED
)

def load_and_prepare_data(file_path, target_variable, selected_features):
    """
    Load data and prepare X and y for modeling.
    """
    data = pd.read_parquet(file_path)
    X = data[selected_features]
    y = data[target_variable]
    
    return X, y

def optimize_xgboost(X, y):
    """
    Optimize XGBoost hyperparameters using Bayesian Optimization.
    """
    def xgb_evaluate(max_depth, learning_rate, n_estimators, min_child_weight, subsample):
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            **MODEL_PARAMS['xgboost']['default_params']
        }
        
        model = xgb.XGBRegressor(**params)
        score = cross_val_score(model, X, y, cv=EVALUATION['cv_folds'], scoring='neg_mean_squared_error').mean()
        return score

    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=MODEL_PARAMS['xgboost']['pbounds'],
        random_state=RANDOM_SEED
    )
    
    optimizer.maximize(init_points=3, n_iter=5)
    return optimizer.max

def optimize_random_forest(X, y):
    """
    Optimize Random Forest hyperparameters using Bayesian Optimization.
    """
    def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            **MODEL_PARAMS['random_forest']['default_params']
        }
        
        model = RandomForestRegressor(**params)
        score = cross_val_score(model, X, y, cv=EVALUATION['cv_folds'], scoring='neg_mean_squared_error').mean()
        return score

    optimizer = BayesianOptimization(
        f=rf_evaluate,
        pbounds=MODEL_PARAMS['random_forest']['pbounds'],
        random_state=RANDOM_SEED
    )
    
    optimizer.maximize(init_points=3, n_iter=5)
    return optimizer.max

def optimize_svr(X, y):
    """
    Optimize SVR hyperparameters using Bayesian Optimization.
    """
    def svr_evaluate(C, epsilon):
        params = {
            'C': C,
            'epsilon': epsilon,
            **MODEL_PARAMS['svr']['default_params']
        }
        model = SVR(**params)
        score = cross_val_score(model, X, y, cv=EVALUATION['cv_folds'], scoring='neg_mean_squared_error').mean()
        return score

    optimizer = BayesianOptimization(
        f=svr_evaluate,
        pbounds=MODEL_PARAMS['svr']['pbounds'],
        random_state=RANDOM_SEED
    )
    
    optimizer.maximize(init_points=3, n_iter=5)
    return optimizer.max

def train_and_save_models(X, y, model_type, output_dir=MODELS_DIR):
    """
    Train models with optimized parameters and save them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Optimize and train XGBoost
    print("Optimizing XGBoost...")
    xgb_params = optimize_xgboost(X, y)
    xgb_model = xgb.XGBRegressor(
        max_depth=int(xgb_params['params']['max_depth']),
        learning_rate=xgb_params['params']['learning_rate'],
        n_estimators=int(xgb_params['params']['n_estimators']),
        min_child_weight=xgb_params['params']['min_child_weight'],
        subsample=xgb_params['params']['subsample'],
        colsample_bytree=0.8,
        objective='reg:squarederror'
    )
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, f'{output_dir}/xgboost_{model_type}.joblib')
    
    # Optimize and train Random Forest
    print("Optimizing Random Forest...")
    rf_params = optimize_random_forest(X, y)
    rf_model = RandomForestRegressor(
        n_estimators=int(rf_params['params']['n_estimators']),
        max_depth=int(rf_params['params']['max_depth']),
        min_samples_split=int(rf_params['params']['min_samples_split']),
        min_samples_leaf=int(rf_params['params']['min_samples_leaf']),
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    joblib.dump(rf_model, f'{output_dir}/random_forest_{model_type}.joblib')
    
    # Optimize and train SVR
    print("Optimizing SVR...")
    svr_params = optimize_svr(X, y)
    svr_model = SVR(
        C=svr_params['params']['C'],
        epsilon=svr_params['params']['epsilon'],
        gamma=svr_params['params']['gamma'],
        kernel='rbf'
    )
    svr_model.fit(X, y)
    joblib.dump(svr_model, f'{output_dir}/svr_{model_type}.joblib')
    
    return {
        'xgboost': xgb_model,
        'random_forest': rf_model,
        'svr': svr_model
    }

if __name__ == "__main__":
    # Import feature selection functions
    from feature_selection import load_data, combine_selected_features
    
    # Get file paths from config
    rent_data_path = DATA_FILES['rent']['output']
    sale_data_path = DATA_FILES['sale']['output']
    
    # Define target variables from config
    rent_target = DATA_FILES['rent']['target']
    sale_target = DATA_FILES['sale']['target']
    
    # Get selected features directly using feature selection
    print("Loading and selecting features for rent data...")
    rent_data = load_data(rent_data_path)
    rent_selected_features = combine_selected_features(rent_data, rent_target)
    
    print("Processing rent data...")
    X_rent, y_rent = load_and_prepare_data(
        rent_data_path, 
        rent_target, 
        list(rent_selected_features)  # Convert set to list
    )
    rent_models = train_and_save_models(X_rent, y_rent, 'rent')
    
    # Process sale data
    print("\nLoading and selecting features for sale data...")
    sale_data = load_data(sale_data_path)
    sale_selected_features = combine_selected_features(sale_data, sale_target)
    
    print("Processing sale data...")
    X_sale, y_sale = load_and_prepare_data(
        sale_data_path, 
        sale_target, 
        list(sale_selected_features)  # Convert set to list
    )
    sale_models = train_and_save_models(X_sale, y_sale, 'sale')
  