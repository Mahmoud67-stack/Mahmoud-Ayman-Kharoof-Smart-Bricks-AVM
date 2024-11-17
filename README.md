# Dubai Real Estate Automated Valuation Model (AVM)
## 📋 Project Overview

The Dubai Real Estate Automated Valuation Model (AVM) is a sophisticated machine learning system designed to estimate property values in the Dubai real estate market. The project leverages advanced AI algorithms and ensemble learning techniques to provide accurate property valuations for both rental and sales properties.


### Key Features

- 🔄 Comprehensive data preprocessing pipeline
- 🎯 Intelligent feature selection using multiple methods
- 🤖 Multiple base models (XGBoost, Random Forest, SVR)
- 🧠 Neural network-based meta-learner for improved accuracy
- 📊 Detailed evaluation metrics and visualization
- 🐳 Docker containerization for easy deployment

## 🏗️ Project Structure

```
notebooks/
├── preprocess.py          # Data preprocessing pipeline
├── feature_selection.py   # Feature selection logic
├── base_models.py         # Base model implementation
├── meta_learner.py        # Meta-learner implementation
├── main.py               # Orchestrates the entire pipeline
├── evaluation.py         # Model evaluation metrics
├── config.py             # Configuration parameters
├── Dockerfile            # Docker container setup
├── requirements.txt      # Python dependencies
└── test.py              # Unit tests

data/
├── sales.csv        # Sales data
└── rentals.csv      # Rentals data
Dockerfile            # Docker container setup
README.md            # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- WSL2 with Ubuntu (for Windows users)
- Docker

### Environment Setup

1. **Install WSL (Windows users)**
```bash
# Update WSL environment
sudo apt update && sudo apt upgrade -y
```

2. **Install Python**
```bash
sudo apt install python3 python3-venv python3-pip -y
```

3. **Install Docker**
```bash
sudo apt install docker-ce docker-ce-cli containerd.io -y
sudo usermod -aG docker $USER
newgrp docker
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd dubai-real-estate-avm
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Build Docker container**
```bash
docker build -t avm-app .
```

## 🔧 Usage

### Running with Python

1. **Start the pipeline**
```bash
python main.py
```

### Running with Docker

1. **Run the container**
```bash
docker run -it avm-app
```

## 🧮 Technical Details

### Data Processing Pipeline

The system processes two main types of real estate data:
- Rental Property Data
- Sales Transaction Data

#### Processing Steps:
1. Load raw data (CSV/Parquet)
2. Clean and standardize
3. Handle missing values
4. Process special columns
5. Engineer features
6. Encode categoricals
7. Scale numericals


### Feature Selection

The project employs multiple feature selection techniques:
- Correlation analysis
- Tree-based feature importance
- Recursive feature elimination
- Univariate feature selection
#### Rental Property Feature Importance
![feature_importance_rent](https://github.com/user-attachments/assets/64a0f099-325c-4bdc-a227-dbcbdad737d6)
#### Sales Property Feature Importance
![feature_importance_sale](https://github.com/user-attachments/assets/cafed8fd-fbb2-491a-9995-5bd2229a7bbf)

### Model Architecture

#### Base Models:
- XGBoost
- Random Forest
- Support Vector Regression (SVR)

#### Meta-Learner:
- Neural network combining base model predictions
- Optimized using Bayesian optimization

### Evaluation Metrics

- Root Mean Square Error (RMSE)
- R² Score
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Explained Variance

## 📊 Visualization Examples

### Feature Importance Analysis
#### Rental Property Feature Importance
![feature_importance_rent](https://github.com/user-attachments/assets/4a3f252b-9c72-4640-9b03-849a335dba9a)
#### Sales Property Feature Importance
![feature_importance_sale](https://github.com/user-attachments/assets/6c269f62-479a-4399-bbec-c722734ae1fa)

### Optimization Progress
#### Random Forest Optimization Progress
#### Rental Property Optimization Progress
![RandomForestRent](https://github.com/user-attachments/assets/e33f6db4-3a01-457b-82e1-b15cf4955f77)
![optimization_Random Forest_rent_20241117_151210](https://github.com/user-attachments/assets/95c57196-a76a-4f3d-9519-1066ead506db)
#### Sales Property Optimization Progress
![RANDOMFORESTSALE](https://github.com/user-attachments/assets/bf2d8703-4e5a-4f0d-89ea-527b5ab3fcc9)
![optimization_Random Forest_sale_20241117_152336](https://github.com/user-attachments/assets/b0a58676-b147-454b-b802-c51e9b8ea754)
#### XGBoost Optimization Progress
#### Rental Property Optimization Progress
![XGBOOSTRENT](https://github.com/user-attachments/assets/c2007a91-cbda-435e-8f81-37e5717b84e0)
![optimization_XGBoost_rent_20241117_150658](https://github.com/user-attachments/assets/d118e737-900b-4510-ae63-fa1bf5818b9c)
#### Sales Property Optimization Progress
![SALEXGGBOOST](https://github.com/user-attachments/assets/8d9f0d33-2fee-45f6-b169-c50454618f41)
![optimization_XGBoost_sale_20241117_152309](https://github.com/user-attachments/assets/065703e0-7faf-4a67-a886-341d37c9f28a)
#### SVR Optimization Progress  
#### Rental Property Optimization Progress
![SVRRENT](https://github.com/user-attachments/assets/07c1427b-2249-4fc7-a760-c667a3b4f262)
![optimization_SVR_rent_20241117_151238](https://github.com/user-attachments/assets/611143e9-8984-4638-a132-6d98119983f2)
#### Sales Property Optimization Progress
![svrsale](https://github.com/user-attachments/assets/c5466d56-8e2b-4993-a6f8-3e1c0b5e1ccb)
![optimization_SVR_sale_20241117_152344](https://github.com/user-attachments/assets/55d48dad-ab98-4ef8-a244-ecb1763d1325)
#### Neural Network Optimization Progress
#### Rental Property Optimization Progress
![optimization_results_rent_20241117_151636](https://github.com/user-attachments/assets/c4dc7670-4bd6-4bd8-9c07-ba4cd3e249bd)
#### Sales Property Optimization Progress
![optimization_results_sale_20241117_152440](https://github.com/user-attachments/assets/e8b90b89-7575-4a5a-a13d-d729fa92d904)


### Residual Analysis
#### Rental Property Residual Analysis
![rent_residuals](https://github.com/user-attachments/assets/100233cb-ae9a-47c1-910a-34f768ec9754)
#### Sales Property Residual Analysis
![sale_residuals](https://github.com/user-attachments/assets/dd548b3d-945b-4d34-8f0b-8ce3e55de39e)


### Actual vs. Predicted Comparison
#### Rental Property Actual vs. Predicted
![rent_actual_vs_predicted](https://github.com/user-attachments/assets/8a95a3ae-c33c-40eb-a00c-de9f034b398b)
#### Sales Property Actual vs. Predicted
![sale_actual_vs_predicted](https://github.com/user-attachments/assets/f5f5f48c-92c1-4912-a34a-32616fe8452e)


### Training History
#### Rental Property Training History
![training_history_rent_20241117_151947](https://github.com/user-attachments/assets/af61883c-aecb-403d-b8f3-5ecb8f3f2c4d)
#### Sales Property Training History
![training_history_sale_20241117_152518](https://github.com/user-attachments/assets/d5e3e06b-2c95-4b50-9aff-edd13e986b2d)


## 🧪 Testing

Run the test suite:
```bash
pytest test.py
```

Tests cover:
- Directory setup
- Preprocessing pipeline
- Model training
- Evaluation metrics
- Error handling
- Full pipeline integration

## 📈 Performance

The system achieves competitive performance through:
- Ensemble learning approach
- Hyperparameter optimization
- Feature engineering
- Meta-learning

## ⚙️ Configuration

All major parameters are configurable through `config.py`:
- Data paths
- Model parameters
- Feature selection settings
- Evaluation metrics
- Logging configuration

## 🚧 Error Handling

The system includes comprehensive error handling:
- Detailed logging
- Error tracking
- Exception management
- Data validation


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Author

Mahmoud Ayman Kharoof

## 🤝 Acknowledgments

- Smart Bricks for the opportunity
- Dubai Land Department for data insights
- Open-source ML community

---

*This project was developed as part of a technical assessment for Smart Bricks.*
