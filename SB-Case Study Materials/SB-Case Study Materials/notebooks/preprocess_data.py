import pandas as pd
import os

def preprocess_data(input_file, output_file, columns_to_remove):
    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}. Skipping preprocessing.")
        return  # Exit the function if the file does not exist

    # Load the dataset
    data = pd.read_csv(input_file)

    # Save the original dataset as Parquet format
    original_parquet_file = output_file.replace('.csv', '.parquet')
    data.to_parquet(original_parquet_file, index=False, engine='pyarrow')
    print(f"Original data has been saved to {original_parquet_file}")

    # Drop columns with all values missing
    data = data.dropna(axis=1, how='all')

    # Drop specific columns
    data = data.drop(columns=[col for col in columns_to_remove if col in data.columns], errors='ignore')

    # Save the cleaned dataset as Parquet format
    cleaned_parquet_file = output_file.replace('.csv', '_cleaned.parquet')
    data.to_parquet(cleaned_parquet_file, index=False, engine='pyarrow')
    print(f"Cleaned data has been processed and saved to {cleaned_parquet_file}")

    # Delete the original CSV file after processing
    if os.path.isfile(input_file):
        os.remove(input_file)
        print(f"Deleted original file: {input_file}")

if __name__ == "__main__":
    # Adjust the path to go up one level to access the data directory
    input_file_1 = os.path.join('..', 'data', 'snp_dld_2024_rents.csv') 
    output_file_1 = os.path.join('..', 'data', 'snp_dld_2024_rents.csv')  

    # Columns to remove for the first CSV
    columns_to_remove_1 = [
        'version_text', 'is_freehold_text',  # Duplicate columns
        'property_type_ar', 'property_subtype_ar',  # Arabic names not needed
        'property_usage_ar',  # Arabic names not needed
        'property_usage_id',  # All values are 0
        'project_name_ar',  # Arabic names not needed
        'area_ar',  # Arabic names not needed
        'area_id',  # All values are 0
        'nearest_landmark_ar',  # Arabic names not needed
        'nearest_metro_ar',  # Arabic names not needed
        'nearest_mall_ar',  # Arabic names not needed
        'master_project_ar',  # Arabic names not needed
        'ejari_property_type_id',  # All values are 0
        'ejari_property_sub_type_id',  # All values are 0
        'entry_id',  # Values don't indicate anything
        'meta_ts'  # Values don't indicate anything
    ]

    # Preprocess the first CSV
    preprocess_data(input_file_1, output_file_1, columns_to_remove_1)

    # Define the input and output file paths for the second CSV
    input_file_2 = os.path.join('..', 'data', 'snp_dld_2024_transactions.csv')  
    output_file_2 = os.path.join('..', 'data', 'snp_dld_2024_transactions.csv')  

    # Columns to remove for the second CSV
    columns_to_remove_2 = [
        'transaction_number',  # Unique identifier for the transaction; not needed for analysis
        'transaction_subtype_id',  # All values are 0
        'property_id',  # All values are 0
        'property_type_ar',  # Arabic property type; not needed for analysis
        'property_type_id',  # All values are 0
        'property_subtype_ar',  # Arabic property subtype; not needed for analysis
        'property_subtype_id',  # All values are 0
        'building_age',  # All values are 0
        'rooms_ar',  # Arabic description of rooms; not needed for analysis
        'project_name_ar',  # Arabic project name; not needed for analysis
        'area_ar',  # Arabic area name; not needed for analysis
        'area_id',  # All values are 0
        'nearest_landmark_ar',  # Arabic nearest landmark; not needed for analysis
        'nearest_metro_ar',  # Nearest metro in Arabic; not needed for analysis
        'nearest_mall_ar',  # Nearest mall in Arabic; not needed for analysis
        'master_project_ar',  # Arabic master project name; not needed for analysis
        'entry_id',  # Unique entry identifier; does not provide meaningful information
        'meta_ts'  # Timestamp metadata; not needed for analysis
    ]

    # Preprocess the second CSV
    preprocess_data(input_file_2, output_file_2, columns_to_remove_2)
