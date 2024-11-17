import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from bayes_opt import BayesianOptimization
from config import META_LEARNER_PARAMS, MODELS_DIR, DATA_DIR, DATA_FILES, PLOTS_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

class MetaLearner:
    def __init__(self, base_models_dir=MODELS_DIR, model_type='rent'):
        """
        Initialize MetaLearner with base models.
        model_type: 'rent' or 'sale'
        """
        self.model_type = model_type
        self.base_models = self._load_base_models(base_models_dir)
        self.scaler = StandardScaler()
        self.meta_model = None
        self.optimization_history = []
        
    def _load_base_models(self, base_models_dir):
        """Load all base models from directory"""
        base_models_dir = os.path.join('SB-Case Study Materials', base_models_dir)
        
        # Load models
        models = {
            'xgboost': joblib.load(os.path.join(base_models_dir, f'xgboost_{self.model_type}.joblib')),
            'random_forest': joblib.load(os.path.join(base_models_dir, f'random_forest_{self.model_type}.joblib')),
            'svr': joblib.load(os.path.join(base_models_dir, f'svr_{self.model_type}.joblib'))
        }
        
        # Get feature names from the random forest model (which typically preserves feature names)
        self.feature_names = models['random_forest'].feature_names_in_
        return models
    
    def _get_base_predictions(self, X):
        """Get predictions from all base models"""
        # Remove target if it exists in X
        X_clean = X.copy()
        if self.model_type == 'rent' and 'annual_amount' in X_clean.columns:
            X_clean = X_clean.drop('annual_amount', axis=1)
        elif self.model_type == 'sale' and 'amount' in X_clean.columns:
            X_clean = X_clean.drop('amount', axis=1)
        
        # Ensure features are in the correct order
        X_ordered = X_clean[self.feature_names]
        predictions = np.column_stack([
            model.predict(X_ordered) for model in self.base_models.values()
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
    
    def optimize_meta_model(self, X, y, validation_split=META_LEARNER_PARAMS['training']['validation_split']):
        """Optimize meta-model hyperparameters using Bayesian Optimization"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Get base model predictions once, outside the evaluation function
        X_train_meta = self._get_base_predictions(X_train)
        X_val_meta = self._get_base_predictions(X_val)
        
        # Scale the predictions once
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
                epochs=5,
                batch_size=128,
                verbose=0
            )
            
            val_loss = history.history['val_loss'][-1]
            self.optimization_history.append({
                'num_hidden_layers': num_hidden_layers,
                'hidden_units': hidden_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'val_loss': val_loss
            })
            
            return -val_loss
        
        # Reduce optimization iterations
        optimizer = BayesianOptimization(
            f=evaluate_meta_model,
            pbounds=META_LEARNER_PARAMS['optimization']['pbounds'],
            random_state=42
        )
        
        optimizer.maximize(
            init_points=3,  # Reduced from default
            n_iter=5       # Reduced from default
        )
        
        # Plot optimization results
        self._plot_optimization_results()
        
        return optimizer.max
    
    def _plot_optimization_results(self):
        """Plot the optimization process results"""
        # Convert optimization history list to DataFrame
        history_df = pd.DataFrame(self.optimization_history)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Hyperparameter Optimization Results - {self.model_type.upper()} Model')
        
        params = ['num_hidden_layers', 'hidden_units', 'dropout_rate', 'learning_rate']
        for ax, param in zip(axes.flat, params):
            sns.scatterplot(
                data=history_df,  # Now using DataFrame instead of list
                x=param,
                y='val_loss',
                ax=ax
            )
            ax.set_title(f'{param.replace("_", " ").title()} vs Validation Loss')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel('Validation Loss')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'optimization_results_{self.model_type}_{timestamp}.png'
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved optimization plot to: {filepath}")
    
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
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train and store history
        history = self.meta_model.fit(
            X_meta, 
            y, 
            epochs=META_LEARNER_PARAMS['training']['epochs'],
            batch_size=128,
            verbose=META_LEARNER_PARAMS['training']['verbose'],
            callbacks=[early_stopping],
            validation_split=0.2  # Add validation split for plotting
        )
        
        # Plot training history
        self._plot_training_history(history)
    
    def _plot_training_history(self, history):
        """Plot the training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Meta-Learner Training History - {self.model_type.upper()} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'training_history_{self.model_type}_{timestamp}.png'
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved training history plot to: {filepath}")
    
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
        
        try:
            # Load data
            data_path = DATA_FILES[model_type]['output']
            
            # Debug prints for data loading
            print(f"Loading data from: {data_path}")
            data = load_data(data_path)
            print(f"Type of loaded data: {type(data)}")
            
            if not isinstance(data, pd.DataFrame):
                print("Warning: load_data() did not return a DataFrame")
                # Try to convert to DataFrame if it's a list of dictionaries
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    raise TypeError("Could not convert data to DataFrame")
            
            # Get target from config
            target = DATA_FILES[model_type]['target']
            
            # Get selected features and convert set to list
            selected_features = list(combine_selected_features(data, target))
            
            # Create X DataFrame with selected features
            X = data[selected_features]
            y = data[target]
            
            # Initialize and train meta-learner
            print(f"Training meta-learner for {model_type}...")
            meta_learner = MetaLearner(MODELS_DIR, model_type)
            meta_learner.fit(X, y)
            meta_learner.save()
            print(f"Meta-learner for {model_type} trained and saved successfully!")
            
        except Exception as e:
            print(f"Error processing {model_type} data: {str(e)}")
            continue