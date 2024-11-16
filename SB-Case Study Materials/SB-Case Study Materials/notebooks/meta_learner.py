import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from bayes_opt import BayesianOptimization

class MetaLearner:
    def __init__(self, base_models_dir, model_type):
        """
        Initialize MetaLearner with base models.
        model_type: 'rent' or 'sale'
        """
        self.model_type = model_type
        self.base_models = self._load_base_models(base_models_dir)
        self.scaler = StandardScaler()
        self.meta_model = None
        
    def _load_base_models(self, base_models_dir):
        """Load all base models from directory"""
        models = {
            'xgboost': joblib.load(f'{base_models_dir}/xgboost_{self.model_type}.joblib'),
            'random_forest': joblib.load(f'{base_models_dir}/random_forest_{self.model_type}.joblib'),
            'svr': joblib.load(f'{base_models_dir}/svr_{self.model_type}.joblib')
        }
        return models
    
    def _get_base_predictions(self, X):
        """Get predictions from all base models"""
        predictions = np.column_stack([
            model.predict(X) for model in self.base_models.values()
        ])
        return predictions
    
    def _create_meta_model(self, num_hidden_layers, hidden_units, dropout_rate, learning_rate):
        """Create neural network meta-model"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(3,)))  # 3 base models
        
        # Hidden layers
        for _ in range(int(num_hidden_layers)):
            model.add(layers.Dense(int(hidden_units), activation='relu'))
            model.add(layers.Dropout(dropout_rate))
            
        # Output layer
        model.add(layers.Dense(1))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        return model
    
    def optimize_meta_model(self, X, y, validation_split=0.2):
        """Optimize meta-model hyperparameters using Bayesian Optimization"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Get base model predictions
        X_train_meta = self._get_base_predictions(X_train)
        X_val_meta = self._get_base_predictions(X_val)
        
        # Scale the predictions
        X_train_meta = self.scaler.fit_transform(X_train_meta)
        X_val_meta = self.scaler.transform(X_val_meta)
        
        def evaluate_meta_model(num_hidden_layers, hidden_units, dropout_rate, learning_rate):
            model = self._create_meta_model(
                num_hidden_layers,
                hidden_units,
                dropout_rate,
                learning_rate
            )
            
            history = model.fit(
                X_train_meta,
                y_train,
                validation_data=(X_val_meta, y_val),
                epochs=20,
                batch_size=64,
                verbose=0
            )
            
            return -history.history['val_loss'][-1]
        
        optimizer = BayesianOptimization(
            f=evaluate_meta_model,
            pbounds={
                'num_hidden_layers': (1, 2),
                'hidden_units': (16, 64),
                'dropout_rate': (0.1, 0.3),
                'learning_rate': (1e-3, 1e-2)
            },
            random_state=42
        )
        
        optimizer.maximize(init_points=3, n_iter=7)
        return optimizer.max
    
    def fit(self, X, y):
        """Train meta-learner with optimized parameters"""
        # Optimize hyperparameters
        print("Optimizing meta-learner hyperparameters...")
        best_params = self.optimize_meta_model(X, y)
        
        # Get base model predictions
        X_meta = self._get_base_predictions(X)
        X_meta = self.scaler.fit_transform(X_meta)
        
        # Create and train final model with best parameters
        self.meta_model = self._create_meta_model(
            num_hidden_layers=int(best_params['params']['num_hidden_layers']),
            hidden_units=int(best_params['params']['hidden_units']),
            dropout_rate=best_params['params']['dropout_rate'],
            learning_rate=best_params['params']['learning_rate']
        )
        
        self.meta_model.fit(X_meta, y, epochs=50, batch_size=64, verbose=1)
        
    def predict(self, X):
        """Make predictions using the meta-learner"""
        X_meta = self._get_base_predictions(X)
        X_meta = self.scaler.transform(X_meta)
        return self.meta_model.predict(X_meta).flatten()
    
    def save(self, output_dir='models'):
        """Save the meta-learner model and scaler"""
        os.makedirs(output_dir, exist_ok=True)
        self.meta_model.save(f'{output_dir}/meta_learner_{self.model_type}.h5')
        joblib.dump(self.scaler, f'{output_dir}/meta_learner_scaler_{self.model_type}.joblib')

if __name__ == "__main__":
    # Import necessary functions from base_models
    from base_models import load_and_prepare_data
    from feature_selection import load_data, combine_selected_features
    
    # Load and process data for both rent and sale
    for model_type in ['rent', 'sale']:
        print(f"\nTraining meta-learner for {model_type} data...")
        
        # Load data
        data_path = os.path.join('..', 'data', 
                               f'snp_dld_2024_{model_type}{"s" if model_type == "rent" else "_transactions"}_cleaned.parquet')
        target = 'annual_amount' if model_type == 'rent' else 'amount'
        
        # Get selected features
        data = load_data(data_path)
        selected_features = combine_selected_features(data, target)
        
        # Prepare data
        X, y = load_and_prepare_data(data_path, target, list(selected_features))
        
        # Initialize and train meta-learner
        meta_learner = MetaLearner('models', model_type)
        meta_learner.fit(X, y)
        meta_learner.save() 