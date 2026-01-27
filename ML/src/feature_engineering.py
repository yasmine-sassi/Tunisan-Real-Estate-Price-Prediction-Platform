"""
Feature Engineering Module
Creates new features and preprocesses real estate data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import TargetEncoder


class FeatureEngineer:
    """Feature engineering and preprocessing for real estate data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Create engineered features and preprocess data
        
        Args:
            df: Input dataframe
            fit_scaler: Whether to fit scaler (True for train, False for test)
        """
        df = df.copy()
        
        # ===== FEATURE ENGINEERING =====
        
        # Surface-based features
        if 'surface' in df.columns and 'rooms' in df.columns:
            df['surface_per_room'] = df['surface'] / (df['rooms'] + 1)
        
        if 'surface' in df.columns and 'bathrooms' in df.columns:
            df['surface_per_bathroom'] = df['surface'] / (df['bathrooms'] + 1)
        
        # Room density
        if 'rooms' in df.columns and 'bathrooms' in df.columns:
            df['rooms_per_bathroom'] = df['rooms'] / (df['bathrooms'] + 1)
        
        # Location features - identify major cities
        if 'city' in df.columns:
            major_cities = ['Tunis', 'Sfax', 'Sousse', 'Ariana', 'Ben Arous', 'Nabeul', 'Bizerte']
            df['is_major_city'] = df['city'].isin(major_cities).astype(int)
        
        # Property type value indicator
        if 'property_type' in df.columns:
            premium_types = ['Villa', 'Maison']
            df['is_premium_type'] = df['property_type'].isin(premium_types).astype(int)
        
        # Transaction type indicator
        if 'transaction' in df.columns:
            df['is_sale'] = (df['transaction'].str.lower() == 'sale').astype(int)
        
        # ===== CATEGORICAL ENCODING =====
        df = self._encode_categoricals(df)
        
        # ===== NUMERIC NORMALIZATION =====
        df = self._normalize_numerics(df, fit_scaler=fit_scaler)
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode all categorical variables"""
        df = df.copy()
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove price (should be numeric)
        exclude_cols = ['price']
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        print(f"  📝 Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                
                # Low cardinality (≤15 unique values): Label Encoding
                if unique_count <= 15:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"     ✓ {col}: Label Encoded ({unique_count} unique values)")
                
                # High cardinality (>15): Target Encoding
                else:
                    if 'price' in df.columns:
                        te = TargetEncoder(cols=[col])
                        df[f'{col}_encoded'] = te.fit_transform(df[col], df['price'])
                        self.target_encoders[col] = te
                        print(f"     ✓ {col}: Target Encoded ({unique_count} unique values)")
                    else:
                        # Fallback to label encoding
                        le = LabelEncoder()
                        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                        print(f"     ✓ {col}: Label Encoded (fallback, {unique_count} unique values)")
        
        # Drop original categorical columns
        df = df.drop(columns=categorical_cols)
        
        return df
    
    def _normalize_numerics(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize numeric columns using StandardScaler"""
        df = df.copy()
        
        # Get numeric columns (excluding price which is target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'price']
        
        print(f"  📊 Normalizing {len(numeric_cols)} numeric columns: {numeric_cols}")
        
        if fit_scaler:
            # Fit scaler on training data
            X_scaled = self.scaler.fit_transform(df[numeric_cols])
            self.is_fitted = True
        else:
            # Use existing fitted scaler on test data
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call create_features with fit_scaler=True first.")
            X_scaled = self.scaler.transform(df[numeric_cols])
        
        # Replace original numeric columns with scaled versions
        df[numeric_cols] = X_scaled
        
        print(f"     ✓ All numeric features normalized (StandardScaler)")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get list of feature column names"""
        exclude = ['price']
        return [col for col in df.columns if col not in exclude]