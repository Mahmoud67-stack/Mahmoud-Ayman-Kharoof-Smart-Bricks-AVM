import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

from notebooks.main import (
    setup_directories,
    run_preprocessing_pipeline,
    train_models,
    evaluate_models
)
from notebooks.config import DATA_FILES, MODELS_DIR, LOGS_DIR

# Fixtures for common test data
@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    return pd.DataFrame({
        'price': [100000, 200000, 300000],
        'area': [50, 75, 100],
        'rooms': [2, 3, 4],
        'location': ['A', 'B', 'C']
    })

@pytest.fixture
def mock_meta_learner():
    """Create a mock meta learner."""
    meta_learner = MagicMock()
    meta_learner.predict.return_value = np.array([150000, 250000, 350000])
    return meta_learner

# Test setup_directories
def test_setup_directories(tmp_path):
    """Test directory creation functionality."""
    with patch('notebooks.config.MODELS_DIR', str(tmp_path / 'models')):
        with patch('notebooks.config.LOGS_DIR', str(tmp_path / 'logs')):
            setup_directories()
            assert os.path.exists(str(tmp_path / 'models'))
            assert os.path.exists(str(tmp_path / 'logs'))

# Test preprocessing pipeline
@pytest.mark.parametrize("data_type", ["rent", "sale"])
def test_run_preprocessing_pipeline(data_type, sample_data):
    """Test preprocessing pipeline for both rent and sale data."""
    with patch('notebooks.preprocess.preprocess_data') as mock_preprocess:
        with patch('notebooks.main.DATA_FILES', {
            data_type: {
                'input': 'test_input.csv',
                'output': 'test_output.csv',
                'columns_to_remove': ['unused_col'],
                'target': 'price'
            }
        }):
            run_preprocessing_pipeline()
            mock_preprocess.assert_called_once()

# Test model training
def test_train_models(sample_data):
    """Test model training pipeline."""
    model_type = 'rent'
    
    # Mock dependencies
    with patch('notebooks.main.load_data', return_value=sample_data) as mock_load:
        with patch('notebooks.main.combine_selected_features', return_value=['area', 'rooms']) as mock_features:
            with patch('notebooks.main.load_and_prepare_data', return_value=(np.array([[1, 2], [3, 4]]), np.array([100, 200]))) as mock_prepare:
                with patch('notebooks.main.train_and_save_models') as mock_train:
                    with patch('notebooks.main.MetaLearner') as mock_meta:
                        # Configure mock meta learner
                        mock_meta_instance = MagicMock()
                        mock_meta.return_value = mock_meta_instance
                        
                        # Run test
                        X, y, meta_learner = train_models(model_type)
                        
                        # Verify calls
                        mock_load.assert_called_once()
                        mock_features.assert_called_once()
                        mock_prepare.assert_called_once()
                        mock_train.assert_called_once()
                        mock_meta_instance.fit.assert_called_once()
                        mock_meta_instance.save.assert_called_once()

# Test model evaluation
def test_evaluate_models(sample_data, mock_meta_learner):
    """Test model evaluation pipeline."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([100, 200, 300])
    model_type = 'rent'
    
    with patch('notebooks.main.evaluate_model', return_value={
        'mae': 10.0,
        'mse': 100.0,
        'r2': 0.95
    }) as mock_evaluate:
        metrics = evaluate_models(X, y, mock_meta_learner, model_type)
        
        # Verify evaluation was called
        mock_evaluate.assert_called_once()
        
        # Check metrics structure
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'r2' in metrics

# Test error handling
def test_main_error_handling():
    """Test error handling in main pipeline."""
    with patch('notebooks.main.run_preprocessing_pipeline', side_effect=Exception("Test error")):
        with pytest.raises(Exception) as exc_info:
            from notebooks.main import main
            main()
        assert "Test error" in str(exc_info.value)

# Integration test
def test_full_pipeline_integration(sample_data):
    """Test full pipeline integration."""
    with patch('notebooks.main.run_preprocessing_pipeline') as mock_preprocess:
        with patch('notebooks.main.train_models', return_value=(np.array([]), np.array([]), MagicMock())) as mock_train:
            with patch('notebooks.main.evaluate_models', return_value={'mae': 10.0}) as mock_evaluate:
                from notebooks.main import main
                main()
                
                # Verify pipeline steps were called
                mock_preprocess.assert_called_once()
                assert mock_train.call_count == 2  # Called for both rent and sale
                assert mock_evaluate.call_count == 2  # Called for both rent and sale 