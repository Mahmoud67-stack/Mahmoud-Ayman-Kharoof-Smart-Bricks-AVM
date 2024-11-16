import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data files
DATA_FILES = {
    'rent': {
        'input': os.path.join(DATA_DIR, 'snp_dld_2024_rents.csv'),
        'output': os.path.join(DATA_DIR, 'snp_dld_2024_rents_cleaned.parquet'),
        'target': 'annual_amount',
        'columns_to_remove': [
            'total_properties', 'contract_amount', 'registration_date', 'ejari_contract_number',
            'version_text', 'is_freehold_text', 'property_type_ar', 'property_subtype_ar',
            'property_usage_ar', 'property_usage_id', 'project_name_ar', 'area_ar', 'area_id',
            'parcel_id', 'property_id', 'land_property_id', 'nearest_landmark_ar',
            'nearest_metro_ar', 'nearest_mall_ar', 'master_project_ar',
            'ejari_property_type_id', 'ejari_property_sub_type_id', 'entry_id',
            'meta_ts', 'req_from', 'req_to'
        ]
    },
    'sale': {
        'input': os.path.join(DATA_DIR, 'snp_dld_2024_transactions.csv'),
        'output': os.path.join(DATA_DIR, 'snp_dld_2024_transactions_cleaned.parquet'),
        'target': 'amount',
        'columns_to_remove': [
            'transaction_size_sqm', 'transaction_number', 'transaction_type_id',
            'property_usage_id', 'transaction_subtype_id', 'property_id',
            'property_type_ar', 'property_type_id', 'property_subtype_ar',
            'property_subtype_id', 'building_age', 'rooms_ar', 'project_name_ar',
            'transaction_subtype_id', 'area_ar', 'area_id', 'nearest_landmark_ar',
            'nearest_metro_ar', 'nearest_mall_ar', 'master_project_ar',
            'entry_id', 'meta_ts', 'parcel_id', 'req_from', 'req_to'
        ]
    }
}

# Model parameters
MODEL_PARAMS = {
    'xgboost': {
        'pbounds': {
            'max_depth': (3, 8),
            'learning_rate': (0.01, 0.2),
            'n_estimators': (50, 300),
            'min_child_weight': (1, 5),
            'subsample': (0.7, 1.0),
        },
        'default_params': {
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror'
        }
    },
    'random_forest': {
        'pbounds': {
            'n_estimators': (50, 300),
            'max_depth': (5, 20),
            'min_samples_split': (2, 10),
        },
        'default_params': {
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': 42
        }
    },
    'svr': {
        'pbounds': {
            'C': (0.1, 50),
            'epsilon': (0.01, 0.5),
        },
        'default_params': {
            'gamma': 'scale',
            'kernel': 'rbf'
        }
    }
}

# Meta-learner parameters
META_LEARNER_PARAMS = {
    'optimization': {
        'pbounds': {
            'num_hidden_layers': (1, 2),
            'hidden_units': (16, 64),
            'dropout_rate': (0.1, 0.3),
            'learning_rate': (1e-3, 1e-2)
        },
        'init_points': 3,
        'n_iter': 7
    },
    'training': {
        'validation_split': 0.2,
        'epochs': 50,
        'batch_size': 64,
        'verbose': 1
    }
}

# Feature selection parameters
FEATURE_SELECTION = {
    'correlation_threshold': 0.8,
    'importance_features_count': 10,
    'rfe_features_count': 5,
    'univariate_features_count': 5
}

# Preprocessing parameters
PREPROCESSING = {
    'missing_categorical_fill': 'Unknown',
    'missing_numerical_strategy': 'median',
    'scaling_method': 'robust'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'filename': os.path.join(LOGS_DIR, 'pipeline.log')
}

# Evaluation parameters
EVALUATION = {
    'metrics': ['RMSE', 'R2', 'MAE', 'MSE', 'MAPE', 'Explained_Variance'],
    'cv_folds': 3,
    'plot_figsize': (10, 6)
}

# Random seed for reproducibility
RANDOM_SEED = 42 