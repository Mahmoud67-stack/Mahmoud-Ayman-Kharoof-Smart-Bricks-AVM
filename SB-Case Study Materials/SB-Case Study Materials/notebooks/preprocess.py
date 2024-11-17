import warnings
import contextlib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib
from config import (
    DATA_FILES,
    PREPROCESSING,
    MODELS_DIR,
    DATA_DIR
)

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(file_path):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    # Ensure file path is absolute using DATA_DIR
    full_path = os.path.join(DATA_DIR, file_path) if not os.path.isabs(file_path) else file_path
    return pd.read_csv(full_path)

def preprocess_rooms(data):
    """
    Preprocess the 'rooms_en' column with persistent LabelEncoder
    """
    if 'rooms_en' in data.columns:
        data['rooms_en'] = data['rooms_en'].replace(['N/A', '(Blank)', '', None], 'No Rooms')
        
        # Updated path
        encoder_path = os.path.join(MODELS_DIR, 'rooms_encoder.joblib')
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data['rooms_category_encoded'] = data['rooms_en'].map(
                dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            ).fillna(-1)
        else:
            label_encoder = LabelEncoder()
            data['rooms_category_encoded'] = label_encoder.fit_transform(data['rooms_en'])
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            joblib.dump(label_encoder, encoder_path)
            
    return data

def preprocess_parking(data):
    """
    Preprocess the 'parking' column by counting the number of parking entries.
    If the parking column already contains counts, it retains those values.
    :param data: Input DataFrame.
    :return: DataFrame with processed 'parking' column.
    """
    if 'parking' in data.columns:
        # Count parking spaces or extract numeric values
        def count_parking(value):
            if pd.isna(value) or value in ['NA', '(Blank)', '']:
                return 0  # Placeholder for missing or invalid data
            elif isinstance(value, (int, float)):  # Check if the value is already a count
                return int(value)  # Keep the count as is
            else:
                # Ensure value is treated as a string before splitting
                value = str(value)
                return len(value.split(','))  # Count the number of entries

        data['parking'] = data['parking'].apply(count_parking)
    return data

def handle_missing_data(data):
    """
    Handle missing data by filling or dropping values based on context.
    :param data: Input DataFrame.
    :return: DataFrame with missing values handled.
    """
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(PREPROCESSING['missing_categorical_fill'], inplace=True)
        else:
            if PREPROCESSING['missing_numerical_strategy'] == 'median':
                data[column].fillna(data[column].median(), inplace=True)
    return data

def feature_engineering(data):
    """
    Perform feature engineering by creating or modifying features.
    :param data: Input DataFrame.
    :return: DataFrame with engineered features.
    """
    if 'contract_start_date' in data.columns and 'contract_end_date' in data.columns:
        # Convert contract start and end dates to datetime format
        data['contract_start_date'] = pd.to_datetime(data['contract_start_date'], errors='coerce')
        data['contract_end_date'] = pd.to_datetime(data['contract_end_date'], errors='coerce')
        # Calculate contract duration in days
        data['contract_duration_days'] = (data['contract_end_date'] - data['contract_start_date']).dt.days
        
        # Convert to timestamp (seconds since epoch) with consistent column names
        data['contract_start_timestamp'] = data['contract_start_date'].astype('int64') // 10**9
        data['contract_end_timestamp'] = data['contract_end_date'].astype('int64') // 10**9
        
        # Drop the original date columns if needed
        data = data.drop(['contract_start_date', 'contract_end_date'], axis=1, errors='ignore')

    if 'transaction_datetime' in data.columns:
        # Convert transaction datetime to timestamp format
        data['transaction_datetime'] = pd.to_datetime(data['transaction_datetime'], errors='coerce').astype('int64') // 10**9  # Convert to seconds

    return data

def preprocess_rent_data(data):
    """
    Preprocess data for rent prediction.
    :param data: Input DataFrame.
    :return: DataFrame with processed features for rent prediction.
    """
    # Handle missing data
    data = handle_missing_data(data)
    
    # Preprocess specific columns
    data = preprocess_rooms(data)
    data = preprocess_parking(data)
    
    # Encode categorical variables
    data = encode_categorical(data, model_type='rent')
    
    # Scale numerical features
    data = scale_numerical(data, model_type='rent', target_variable='annual_amount')
    
    return data

def preprocess_sale_data(data):
    """
    Preprocess data for sale prediction.
    :param data: Input DataFrame.
    :return: DataFrame with processed features for sale prediction.
    """
    # Handle missing data
    data = handle_missing_data(data)
    
    # Preprocess specific columns
    data = preprocess_rooms(data)
    data = preprocess_parking(data)
    
    # Encode categorical variables
    data = encode_categorical(data, model_type='sale')
    
    # Scale numerical features
    data = scale_numerical(data, model_type='sale', target_variable='amount')
    
    return data

def encode_categorical(data, model_type):
    """
    Encode categorical variables with persistent LabelEncoders.
    """
    encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # Skip if no categorical columns
    if len(categorical_columns) == 0:
        return data
        
    for column in categorical_columns:
        try:
            encoder_path = os.path.join(MODELS_DIR, f'{model_type}_{column}_encoder.joblib')
            
            if os.path.exists(encoder_path):
                label_encoder = joblib.load(encoder_path)
                # Handle potential missing categories
                data[column] = data[column].astype(str)  # Convert to string to ensure compatibility
                data[column] = data[column].map(
                    dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                ).fillna(-1)
            else:
                label_encoder = LabelEncoder()
                data[column] = data[column].astype(str)  # Convert to string to ensure compatibility
                data[column] = label_encoder.fit_transform(data[column])
                os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
                joblib.dump(label_encoder, encoder_path)
                
            encoders[column] = label_encoder
        except Exception as e:
            print(f"Error encoding column {column}: {str(e)}")
            # Skip problematic column but continue processing
            continue
            
    return data

def scale_numerical(data, model_type, target_variable=None):
    """
    Scale numerical features with persistent RobustScaler, excluding the target variable.
    """
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Skip if no numerical columns or all columns are target variables
    if not numerical_cols or (target_variable and len(numerical_cols) == 1 and target_variable in numerical_cols):
        return data
        
    if target_variable and target_variable in numerical_cols:
        numerical_cols.remove(target_variable)
        
    scaler_path = os.path.join(MODELS_DIR, f'{model_type}_robust_scaler.joblib')
    
    try:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            # Ensure all features exist in current dataset
            common_cols = [col for col in numerical_cols if col in data.columns]
            if common_cols:
                data[common_cols] = scaler.transform(data[common_cols])
        else:
            if PREPROCESSING['scaling_method'] == 'robust':
                scaler = RobustScaler()
                if numerical_cols:
                    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                    joblib.dump(scaler, scaler_path)
    except Exception as e:
        print(f"Error in scaling numerical features: {str(e)}")
        # Return unscaled data if scaling fails
        return data

    return data

def preprocess_data(input_file, output_file, columns_to_remove, model_type):
    """
    Main preprocessing function with error handling
    Workflow:
    1. Read CSV or parquet from data directory
    2. Convert to parquet (raw) if from CSV
    3. Process the data
    4. Save processed parquet
    5. Delete original CSV if it exists
    """
    try:
        model_config = DATA_FILES[model_type]
        
        # Define all file paths
        input_csv = os.path.join(DATA_DIR, input_file)
        raw_parquet = input_csv.replace('.csv', '.parquet')
        processed_parquet = os.path.join(DATA_DIR, model_config['output'])

        # Check for input files
        if os.path.isfile(input_csv):
            print(f"Reading CSV file: {input_csv}")
            data = pd.read_csv(input_csv)
            
            # Save raw parquet
            print(f"Converting to raw parquet: {raw_parquet}")
            os.makedirs(os.path.dirname(raw_parquet), exist_ok=True)
            data.to_parquet(raw_parquet, index=False, engine='pyarrow')
            
            # Clean up CSV after conversion
            print(f"Cleaning up: Removing original CSV file")
            os.remove(input_csv)
        elif os.path.isfile(raw_parquet):
            print(f"Reading existing parquet file: {raw_parquet}")
            data = pd.read_parquet(raw_parquet)
        else:
            raise FileNotFoundError(f"Neither input CSV ({input_csv}) nor parquet ({raw_parquet}) file exists.")

        print(f"\nStarting preprocessing for {model_type} data...")
        
        # Step 3: Process the data
        print("Processing data...")
        data = data.dropna(axis=1, how='all')
        data = data.drop(columns=[col for col in columns_to_remove if col in data.columns], errors='ignore')
        data = handle_missing_data(data)
        data = preprocess_rooms(data)
        data = preprocess_parking(data)
        data = feature_engineering(data)
        data = encode_categorical(data, model_type=model_type)
        data = scale_numerical(data, model_type=model_type, target_variable=model_config['target'])

        # Step 4: Save processed parquet
        print(f"Saving processed parquet: {processed_parquet}")
        os.makedirs(os.path.dirname(processed_parquet), exist_ok=True)
        data.to_parquet(processed_parquet, index=False, engine='pyarrow')

        print(f"\nSuccessfully processed {model_type} data:")
        print(f"Number of rows: {len(data)}")
        print(f"Number of columns: {len(data.columns)}")
        print(f"Raw parquet saved to: {raw_parquet}")
        print(f"Processed parquet saved to: {processed_parquet}")
        print(f"Original CSV removed: {input_csv}")

    except Exception as e:
        print(f"Error processing {model_type} data: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting preprocessing pipeline...")
    
    # Process rent data
    print("\n=== Processing Rent Data ===")
    preprocess_data(
        DATA_FILES['rent']['input'],
        DATA_FILES['rent']['output'],
        DATA_FILES['rent']['columns_to_remove'],
        model_type='rent'
    )

    # Process sale data
    print("\n=== Processing Sale Data ===")
    preprocess_data(
        DATA_FILES['sale']['input'],
        DATA_FILES['sale']['output'],
        DATA_FILES['sale']['columns_to_remove'],
        model_type='sale'
    )
    
    print("\nPreprocessing pipeline completed!")
