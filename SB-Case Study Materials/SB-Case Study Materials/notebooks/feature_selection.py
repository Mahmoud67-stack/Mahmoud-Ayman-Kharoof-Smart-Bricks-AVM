import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
import matplotlib.pyplot as plt
import os
from config import (
    DATA_FILES, FEATURE_SELECTION, PLOTS_DIR,
    RANDOM_SEED, EVALUATION
)

def load_data(file_path):
    """
    Load data from a Parquet file.
    :param file_path: Path to the Parquet file.
    :return: DataFrame containing the loaded data.
    """
    return pd.read_parquet(file_path)

def correlation_analysis(data, threshold=FEATURE_SELECTION['correlation_threshold']):
    """
    Perform correlation analysis to identify highly correlated features.
    :param data: Input DataFrame.
    :param threshold: Correlation threshold from config.
    :return: List of features to drop.
    """
    # Exclude datetime columns from correlation analysis
    numeric_data = data.select_dtypes(exclude=['datetime64'])
    
    corr_matrix = numeric_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

def feature_importance_tree_based(data, target_variable):
    """
    Determine feature importance using a RandomForestRegressor.
    """
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Balanced parameters for both speed and accuracy
    model = RandomForestRegressor(
        n_estimators=100,         # Keep good number of trees for stability
        max_depth=20,             # Reasonable depth
        min_samples_leaf=20,      # Prevent overfitting but maintain detail
        max_features=0.7,         # Use 70% of features for each split
        n_jobs=-1,               # Use all CPU cores
        random_state=RANDOM_SEED
    )
    print("Fitting Random Forest model...")
    model.fit(X, y)
    
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.sort_values(ascending=False)

def plot_and_save_feature_importance(feature_importances, model_type):
    """
    Plot and save feature importance as a PNG file.
    :param feature_importances: Series of feature importances.
    :param model_type: Type of model ('rent' or 'sale').
    """
    plt.figure(figsize=EVALUATION['plot_figsize'])
    feature_importances.plot(kind='bar')
    plt.title(f'Feature Importance for {model_type.capitalize()} Data')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'feature_importance_{model_type}.png'))
    plt.close()

def recursive_feature_elimination(data, target_variable, n_features_to_select=FEATURE_SELECTION['rfe_features_count']):
    """
    Perform Recursive Feature Elimination (RFE) to select features.
    """
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Use a slightly simplified forest for RFE, but still maintain quality
    model = RandomForestRegressor(
        n_estimators=50,          # Reduced trees but still enough for stability
        max_depth=15,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    
    print("Performing RFE...")
    rfe = RFE(model, n_features_to_select=n_features_to_select, step=2)  # Small step size for accuracy
    rfe.fit(X, y)
    
    return X.columns[rfe.support_]

def univariate_feature_selection(data, target_variable, k=FEATURE_SELECTION['univariate_features_count']):
    """
    Perform univariate feature selection.
    :param data: Input DataFrame.
    :param target_variable: Target variable for prediction.
    :param k: Number of top features to select.
    :return: List of selected features.
    """
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()]
    return selected_features

def combine_selected_features(data, target_variable):
    """
    Combine features selected by different methods.
    """
    print("\nStarting correlation analysis...")
    corr_features = correlation_analysis(data)
    print(f"Found {len(corr_features)} highly correlated features")
    
    print("\nStarting feature importance analysis...")
    importance_features = feature_importance_tree_based(data, target_variable)
    print("Top 10 important features:", importance_features.head(10).index.tolist())
    
    print("\nStarting RFE analysis...")
    rfe_features = recursive_feature_elimination(data, target_variable)
    print("RFE selected features:", rfe_features.tolist())
    
    print("\nStarting univariate feature selection...")
    univariate_features = univariate_feature_selection(data, target_variable)
    print("Univariate selected features:", univariate_features.tolist())
    
    # Combine features with more weight to RF importance
    combined_features = set(importance_features.head(FEATURE_SELECTION['importance_features_count']).index)
    combined_features = combined_features.union(rfe_features, univariate_features)
    combined_features = combined_features.difference(corr_features)
    
    return combined_features

if __name__ == "__main__":
    print("Loading data...")
    rent_data_path = DATA_FILES['rent']['output']
    sale_data_path = DATA_FILES['sale']['output']
    
    rent_data = load_data(rent_data_path)
    sale_data = load_data(sale_data_path)
    
    # Define target variables from config
    rent_target = DATA_FILES['rent']['target']
    sale_target = DATA_FILES['sale']['target']
    
    print(f"\nProcessing rent data (total rows: {len(rent_data)})...")
    rent_selected_features = combine_selected_features(rent_data, rent_target)
    print("\nFinal selected features for rent data:", sorted(rent_selected_features))
    
    # Plot feature importance for rent data
    print("\nGenerating feature importance plot for rent data...")
    rent_importance = feature_importance_tree_based(rent_data, rent_target)
    plot_and_save_feature_importance(rent_importance, 'rent')
    
    print(f"\nProcessing sale data (total rows: {len(sale_data)})...")
    sale_selected_features = combine_selected_features(sale_data, sale_target)
    print("\nFinal selected features for sale data:", sorted(sale_selected_features))
    
    # Plot feature importance for sale data
    print("\nGenerating feature importance plot for sale data...")
    sale_importance = feature_importance_tree_based(sale_data, sale_target)
    plot_and_save_feature_importance(sale_importance, 'sale')