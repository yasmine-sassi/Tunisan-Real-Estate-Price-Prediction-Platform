"""
Tunisian Real Estate Price Prediction - ML Pipeline
Main training script with MLflow tracking
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import custom modules
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))


def get_next_experiment_version():
    """Get the next version number for experiments"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        # Filter experiments matching pattern: tunisian_real_estate_prediction_vX
        version_numbers = []
        for exp in experiments:
            exp_name = exp.name
            if exp_name.startswith('tunisian_real_estate_prediction_v'):
                try:
                    version_str = exp_name.replace('tunisian_real_estate_prediction_v', '')
                    version_num = int(version_str)
                    version_numbers.append(version_num)
                except ValueError:
                    pass
        
        # Get next version
        next_version = max(version_numbers) + 1 if version_numbers else 1
        return next_version
    
    except Exception as e:
        print(f"⚠️  Could not retrieve existing experiments: {e}")
        print("   Starting with version 1")
        return 1


class RealEstatePipeline:
    """Main ML Pipeline for Real Estate Price Prediction"""
    
    def __init__(self, experiment_name: str = None):
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Create versioned experiment name if not provided
        if experiment_name is None:
            version = get_next_experiment_version()
            experiment_name = f'tunisian_real_estate_prediction_v{version}'
            print(f"📊 Created new experiment: {experiment_name}")
        
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'lasso': Lasso(alpha=1.0, random_state=RANDOM_STATE),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                verbosity=0
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                verbose=-1
            )
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV"""
        print(f"📂 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} rows with {df.shape[1]} columns")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        return df
    
    def prepare_data(self, df: pd.DataFrame, transaction: str = 'all'):
        """Prepare data for training"""
        print(f"\n🔧 Preparing data for transaction type: {transaction}")
        
        # Filter by transaction type if specified
        if transaction != 'all':
            df = df[df['transaction'].str.lower() == transaction.lower()].copy()
            print(f"  ✓ Filtered to {len(df)} {transaction} properties")
        
        # Apply feature engineering
        print(f"  ✓ Engineering and preprocessing features...")
        df_featured = self.feature_engineer.create_features(df, fit_scaler=True)
        
        print(f"  ✓ Created {df_featured.shape[1]} features")
        
        # Separate features and target
        target_col = 'price'
        if target_col not in df_featured.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df_featured.drop(columns=[target_col])
        y = df_featured[target_col]
        
        # Remove any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"  ✓ Final dataset: {len(X)} samples with {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        return X_train, X_test, y_train, y_test, X_train.columns.tolist()
    
    def train_and_evaluate(self, model_name: str, model, X_train, X_test, y_train, y_test, feature_names):
        """Train a model and log results to MLflow"""
        print(f"\n🤖 Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            metrics = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_r2': r2_score(y_test, y_pred_test),
            }
            
            # Log parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for key, value in params.items():
                    if key not in ['random_state']:
                        try:
                            mlflow.log_param(key, value)
                        except:
                            pass
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature information
            mlflow.log_param("n_features", len(feature_names))
            
            # Print results
            print(f"  ✓ Train MAE: {metrics['train_mae']:,.0f} | R²: {metrics['train_r2']:.4f}")
            print(f"  ✓ Test MAE:  {metrics['test_mae']:,.0f} | R²: {metrics['test_r2']:.4f}")
            print(f"  ✓ Test RMSE: {metrics['test_rmse']:,.0f}")
            
            return metrics
    
    def run_experiment(self, data_path: str, transaction: str = 'all'):
        """Run complete experiment"""
        print(f"\n{'='*70}")
        print(f"🚀 Starting ML Experiment: {transaction.upper()}")
        print(f"{'='*70}")
        
        # Load and prepare data
        df = self.load_data(data_path)
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(df, transaction)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Price range: {y_train.min():,.0f} - {y_train.max():,.0f}")
        
        # Train all models
        results = {}
        for model_name, model in self.models.items():
            try:
                metrics = self.train_and_evaluate(
                    model_name, model, X_train, X_test, y_train, y_test, feature_names
                )
                results[model_name] = metrics
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {str(e)}")
        
        # Print summary
        if results:
            print(f"\n{'='*70}")
            print("📈 Experiment Summary")
            print(f"{'='*70}")
            
            results_df = pd.DataFrame(results).T
            results_df = results_df.sort_values('test_r2', ascending=False)
            
            print("\n{:<20} {:>12} {:>12} {:>10}".format("Model", "Test MAE", "Test RMSE", "Test R²"))
            print("-" * 56)
            for idx, row in results_df.iterrows():
                print("{:<20} {:>12,.0f} {:>12,.0f} {:>10.4f}".format(
                    idx, row['test_mae'], row['test_rmse'], row['test_r2']
                ))
            
            best_model = results_df.index[0]
            best_r2 = results_df.loc[best_model, 'test_r2']
            best_mae = results_df.loc[best_model, 'test_mae']
            
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   R² Score: {best_r2:.4f}")
            print(f"   MAE: {best_mae:,.0f}")
        
        return results


def main():
    """Main execution"""
    # Create pipeline with auto-versioned experiment
    pipeline = RealEstatePipeline()
    
    # Check if data exists
    data_path = Path("data/processed/final_real_estate_dataset.csv")
    
    if not data_path.exists():
        print("⚠️  No data found!")
        print(f"   Expected path: {data_path.absolute()}")
        return
    
    # Run experiments for both transaction types
    print("\n" + "="*70)
    print("🏠 TUNISIAN REAL ESTATE PRICE PREDICTION")
    print(f"📊 Experiment: {pipeline.experiment_name}")
    print("="*70)
    
    # Train on sale properties
    pipeline.run_experiment(str(data_path), transaction='sale')
    
    # Train on rent properties
    pipeline.run_experiment(str(data_path), transaction='rent')
    
    # Train on all properties
    pipeline.run_experiment(str(data_path), transaction='all')
    
    print("\n" + "="*70)
    print("✅ All experiments completed!")
    print(f"📊 Experiment: {pipeline.experiment_name}")
    print(f"📊 View results at: {MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()