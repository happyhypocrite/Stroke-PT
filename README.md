## Stroke-XG

An automated machine learning pipeline for predicting stroke patient outcomes using XGBoost regression, Bayesian hyperparameter optimization, cross-validation, and recursive feature elimination. Designed for clinical researchers analyzing longitudinal stroke recovery datasets.

### Project Overview

This pipeline implements a comprehensive ML workflow to predict stroke patient cognitive outcomes using clinical and demographic data. The system automates data preprocessing, hyperparameter optimization, feature selection, and model evaluation with proper train/validation/test splits for unbiased performance assessment.

### Core Components
1. **Automated data preprocessing** with type checking and categorical encoding
2. **Configurable pipeline setup via ModelConfig dataclass**
3. **Bayesian hyperparameter optimization** using Tree-structured Parzen Estimator (TPE)
4. **Cross-validation** for robust model selection and metric reporting
5. **Recursive Feature Elimination (RFE)** to identify the most predictive features, with automatic selection of the best RFE iteration based on validation metrics
6. **Comprehensive results tracking** with iteration-by-iteration statistics and Excel output

### Features

#### Data Processing
- Label encoding for categorical features
- User-defined column removal for irrelevant features
- Type checking and conversion for numeric columns

#### Model Training
- XGBoostRegressor with full hyperparameter space optimization
- Cross-validation for all model selection and metric reporting
- Configurable loss metrics (MAE or RMSE)

#### Feature Selection
- Recursive Feature Elimination based on feature importance scores
- Zero-importance feature removal with configurable minimum feature thresholds
- Iterative model retraining after each elimination round
- Feature importance tracking across all iterations
- Automatic selection of the best RFE iteration (lowest MAE, highest R²)

#### Results & Storage
- Excel output with separate sheets for each RFE iteration
- Model persistence in JSON format for future use
- Comprehensive statistics including validation and test performance
- Hyperparameter logging for reproducibility

### Requirements

- Python 3.12+
- See pyproject.toml for complete dependency list

#### Key Dependencies
- xgboost>=2.1.4
- hyperopt>=0.2.7
- scikit-learn>=1.6.1
- pandas>=2.3.2
- numpy>=2.0.2
- alive-progress>=3.3.0
- openpyxl>=3.1.5

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Stroke-XG.git
cd Stroke-XG

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e .
```

### Configuration

All pipeline settings are managed via the ModelConfig object in main.py.

#### Example:
```python
from src.model import ModelConfig, ModelPipeline

config = ModelConfig(
    csv_path="path/to/your/stroke_data.csv",
    save_dir="path/to/output/directory",
    index_col="Study_ID",
    target_feature_y="Normed_MoCA_V5",
    columns_to_drop=["unwanted_col1", "unwanted_col2"],
    trials_param_eval=100,        # Hyperparameter optimization trials
    recursive_trials=True,        # Enable recursive feature elimination
    min_features_per_sample=3,    # Minimum features to retain
    trials_loss_metric="mae"      # Loss metric for optimization
)

pipeline = ModelPipeline(config)
pipeline.run()
```


#### Configuration Parameters

| Parameter              | Type        | Default | Description |
|------------------------|------------|---------|-------------|
| csv_path               | str        | -       | Path to input CSV file |
| save_dir               | str        | -       | Directory for saving outputs |
| index_col              | str        | -       | Column to use as DataFrame index |
| target_feature_y       | str        | -       | Target variable for prediction |
| columns_to_drop        | List[str]  | []      | Columns to exclude from modeling |
| seed                   | int        | 42      | Random seed for reproducibility |
| test_size              | float      | 0.2     | Proportion for test/validation splits |
| trials_param_eval      | int        | 100     | Number of hyperparameter optimization trials |
| trials_loss_metric     | str        | "mae"  | Loss metric ("mae" or "rmse") |
| recursive_trials       | bool       | True    | Enable recursive feature elimination |
| min_features_per_sample| int        | 3       | Minimum features to retain during RFE |
| n_validation_folds     | int        | 3       | Number of folds for cross-validation |

#### Config Tips
> Recursive Feature Elimination can be disabled by setting recursive_trials=False for faster training when feature selection is not needed.
> Increase trials_param_eval for more thorough hyperparameter optimization at the cost of longer runtime.
> Adjust min_features_per_sample based on your dataset size - smaller datasets may need higher minimum thresholds.

### Usage

Run the pipeline from the command line:
```bash
python src/main.py
```

### Pipeline Stages

### 1. Data Preprocessing
```
CSV Loading → Type Checking → Column Removal → Missing Value Handling → Label Encoding
```

### 2. Model Optimization
```
Train/Val/Test Split → Hyperparameter Search → Best Model Selection
```

### 3. Recursive Feature Elimination
```
Initial Training → Feature Importance → Zero-Importance Removal → Retrain → Repeat
```

### 4. Final Evaluation
```
Final Model Training → Test Set Evaluation → Results Storage → Model Persistence
```

## Output Files

The pipeline generates several output files in your specified `save_dir`:

- **`XGBoost_Model_Results{target_feature_y}.xlsx`** - Comprehensive results with multiple sheets:
  - `Iteration_1`, `Iteration_2`, etc. - RFE iteration statistics
  - `Validation_Model` - Final validation performance and hyperparameters
  - `Final_Model` - Combined validation and test performance
  
- **`XGB_Final_Model.json`** - Trained XGBoost model for future predictions

## Methodology

### Hyperparameter Optimization
- **Algorithm**: Tree-structured Parzen Estimator (TPE)
- **Search Space**: 8 key XGBoost hyperparameters
- **Validation**: Early stopping on validation set
- **Objective**: Minimize MAE or RMSE on validation data

### Feature Selection Strategy
- **Method**: Recursive elimination of zero-importance features
- **Safety**: Maintains minimum feature threshold
- **Retraining**: Full model retraining after each elimination round

### Model Evaluation
- **Train Set**: Model fitting and hyperparameter optimization
- **Validation Set**: Feature elimination and hyperparameter selection
- **Test Set**: Final unbiased performance evaluation
- **Metrics**: MAE, RMSE, and R² score

## Advanced Usage

### Custom Feature Selection
```python
# Disable recursive feature elimination
config = ModelConfig(
    # ... other parameters ...
    recursive_trials=False,  # Skip RFE, use all features
)
```

### Different Loss Metrics
```python
# Optimize for RMSE instead of MAE
config = ModelConfig(
    # ... other parameters ...
    trials_loss_metric="rmse",
)
```

### Extended Hyperparameter Search
```python
# More thorough hyperparameter search
config = ModelConfig(
    # ... other parameters ...
    trials_param_eval=500,  # 5x more trials
)
```

## Performance Monitoring

The pipeline provides real-time feedback:
- **Hyperparameter optimization** progress with live loss values
- **Feature elimination** progress with animated progress bars  
- **Iteration statistics** printed during RFE process
- **Final test results** with comprehensive metrics

## Contributing

1. Fork the repository
2. Create a feature branch 
3. Commit your changes 
4. Push to the branch 
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This pipeline was developed specifically for stroke research applications. While the core ML components are generalizable, the data preprocessing and feature selection strategies are optimized for clinical stroke datasets.
