"""
Tunisian Real Estate Price Prediction - MLflow Model Comparison
Uses existing preprocessed data (X_train, X_test, y_train, y_test)
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path

# Import models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
RANDOM_STATE = 42

# Data paths
BASE_DIR = Path(__file__).resolve().parent
DATA_VERSION = os.getenv("DATA_VERSION", "v2")
DATA_DIR = BASE_DIR / "data" / f"prepared_{DATA_VERSION}"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data" / "prepared"
MODELS_DIR = BASE_DIR / "models"


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def load_preprocessed_data(transaction_type):
    """
    Load preprocessed data for rent or sale
    
    Args:
        transaction_type: 'rent' or 'sale'
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\n📂 Loading {transaction_type.upper()} data...")
    
    data_path = Path(DATA_DIR) / transaction_type
    
    X_train = np.load(data_path / "X_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_test = np.load(data_path / "y_test.npy")
    
    print(f"✅ Loaded {transaction_type.upper()} data:")
    print(f"   Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    print(f"   Price range: {y_train.min():.0f} - {y_train.max():.0f} TND")
    
    return X_train, X_test, y_train, y_test


def get_models():
    """Define all models to compare"""
    return {
        'Linear_Regression': LinearRegression(),
        
        'Ridge_Regression': Ridge(
            alpha=3.0,
            random_state=RANDOM_STATE
        ),
        
        'Lasso_Regression': Lasso(
            alpha=0.0005,
            random_state=RANDOM_STATE,
            max_iter=30000
        ),
        
        'Decision_Tree': DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=4,
            random_state=RANDOM_STATE
        ),
        
        'Random_Forest': RandomForestRegressor(
            n_estimators=600,
            max_depth=18,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        'Gradient_Boosting': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=6,
            min_samples_leaf=3,
            subsample=0.85,
            random_state=RANDOM_STATE
        ),
        
        'XGBoost': XGBRegressor(
            n_estimators=800,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        
        'LightGBM': LGBMRegressor(
            n_estimators=800,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    }


def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, transaction_type):
    """
    Train a model and log everything to MLflow
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train, X_test, y_train, y_test: Preprocessed data
        transaction_type: 'rent' or 'sale'
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n  🤖 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{transaction_type}_{model_name}"):
        # Log transaction type as tag
        mlflow.set_tag("transaction_type", transaction_type)
        mlflow.set_tag("model_type", model_name)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'train_mape': calculate_mape(y_train, y_train_pred)
        }
        
        test_metrics = {
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mape': calculate_mape(y_test, y_test_pred)
        }
        
        all_metrics = {**train_metrics, **test_metrics}
        
        # Calculate overfitting gap
        overfitting_gap = train_metrics['train_r2'] - test_metrics['test_r2']
        all_metrics['overfitting_gap'] = overfitting_gap
        
        # Log model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                # Skip non-serializable parameters
                if key not in ['random_state', 'n_jobs', 'verbose', 'verbosity']:
                    try:
                        mlflow.log_param(key, value)
                    except Exception:
                        pass
        
        # Log dataset information
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("price_range_min", float(y_train.min()))
        mlflow.log_param("price_range_max", float(y_train.max()))
        
        # Log all metrics
        mlflow.log_metrics(all_metrics)
        
        # Log the model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"{transaction_type}_{model_name}"
        )
        
        # Print results
        print(f"     Train R²: {train_metrics['train_r2']:.4f} | MAE: {train_metrics['train_mae']:.2f}")
        print(f"     Test R²:  {test_metrics['test_r2']:.4f} | MAE: {test_metrics['test_mae']:.2f}")
        print(f"     Test RMSE: {test_metrics['test_rmse']:.2f} | MAPE: {test_metrics['test_mape']:.2f}%")
        print(f"     Overfitting gap: {overfitting_gap:.4f}")
        
        return all_metrics, model


def run_experiment(transaction_type, experiment_name):
    """
    Run complete experiment for a transaction type
    
    Args:
        transaction_type: 'rent' or 'sale'
        experiment_name: MLflow experiment name
    
    Returns:
        DataFrame with results, dict of trained models
    """
    print(f"\n{'='*80}")
    print(f"🏠 Running Experiment: {transaction_type.upper()}")
    print(f"{'='*80}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(transaction_type)
    
    # Get models
    models = get_models()
    
    # Train all models
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        try:
            metrics, trained_model = train_and_log_model(
                model_name, model, X_train, X_test, y_train, y_test, transaction_type
            )
            
            # Store results
            results.append({
                'model': model_name,
                **metrics
            })
            
            trained_models[model_name] = trained_model
            
        except Exception as e:
            print(f"     ❌ Error training {model_name}: {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 {transaction_type.upper()} - Results Summary")
    print(f"{'='*80}")
    print(results_df[['model', 'test_r2', 'test_mae', 'test_rmse', 'test_mape', 'overfitting_gap']].to_string(index=False))
    
    # Find best model
    best_model_name = results_df.iloc[0]['model']
    best_r2 = results_df.iloc[0]['test_r2']
    best_mae = results_df.iloc[0]['test_mae']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {best_r2:.4f}")
    print(f"   MAE: {best_mae:,.2f} TND")
    
    return results_df, trained_models


def main():
    """Main execution"""
    print(f"\n{'='*80}")
    print("🏠 TUNISIAN REAL ESTATE PRICE PREDICTION - MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"📂 Data Directory: {DATA_DIR}")
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"real_estate_model_comparison_{timestamp}"
    
    print(f"📊 Experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Get experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"📁 Experiment ID: {experiment.experiment_id}")
    
    # Run experiments for both transaction types
    all_results = {}
    
    # RENT experiment
    rent_results, rent_models = run_experiment('rent', experiment_name)
    all_results['rent'] = rent_results
    
    # SALE experiment
    sale_results, sale_models = run_experiment('sale', experiment_name)
    all_results['sale'] = sale_results
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n🏆 BEST MODELS:")
    print(f"   RENT: {rent_results.iloc[0]['model']} (R² = {rent_results.iloc[0]['test_r2']:.4f})")
    print(f"   SALE: {sale_results.iloc[0]['model']} (R² = {sale_results.iloc[0]['test_r2']:.4f})")
    
    print(f"\n📊 View results in MLflow UI:")
    print(f"   {MLFLOW_TRACKING_URI}")
    print(f"   Experiment: {experiment_name}")
    print(f"   Experiment ID: {experiment.experiment_id}")
    
    # Save results to CSV
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rent_results.to_csv(output_dir / "mlflow_rent_results.csv", index=False)
    sale_results.to_csv(output_dir / "mlflow_sale_results.csv", index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   {output_dir / 'mlflow_rent_results.csv'}")
    print(f"   {output_dir / 'mlflow_sale_results.csv'}")
    
    print(f"\n{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    results = main()