import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

class FashionDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for fashion retail data.
    Handles categorical encoding, numerical scaling, and feature engineering.
    """
    
    def __init__(self):
        self.categorical_cols = ['Category', 'Subcategory', 'Division', 'Gender', 'Size', 
                                'Color', 'Seasonality', 'Material', 'Fit', 'Price_Tier', 
                                'Inventory_Status', 'Product_Age_Category']
        self.numerical_cols = ['Price', 'Inventory', 'Days_Since_Launch']
        self.high_cardinality_cols = ['Subcategory', 'Size', 'Color']  # >10 unique values
        self.low_cardinality_cols = [col for col in self.categorical_cols 
                                    if col not in self.high_cardinality_cols]
        
        self.label_encoders = {}
        self.onehot_encoder = None
        self.numerical_scaler = None
        self.preprocessor = None
        self.feature_names = []
        
    def analyze_features(self, df):
        """Analyze feature characteristics for preprocessing decisions."""
        print("Analyzing features for preprocessing...")
        
        feature_analysis = {}
        
        for col in self.categorical_cols:
            unique_count = df[col].nunique()
            most_common_freq = df[col].value_counts().iloc[0] / len(df)
            
            feature_analysis[col] = {
                'type': 'categorical',
                'unique_count': unique_count,
                'most_common_freq': most_common_freq,
                'cardinality': 'high' if unique_count > 15 else 'medium' if unique_count > 5 else 'low',
                'recommended_encoding': 'target' if unique_count > 15 else 'onehot'
            }
            
            print(f"{col}: {unique_count} unique values, "
                  f"cardinality: {feature_analysis[col]['cardinality']}, "
                  f"encoding: {feature_analysis[col]['recommended_encoding']}")
        
        for col in self.numerical_cols:
            skewness = df[col].skew()
            range_val = df[col].max() - df[col].min()
            
            feature_analysis[col] = {
                'type': 'numerical',
                'skewness': skewness,
                'range': range_val,
                'transformation_needed': abs(skewness) > 1,
                'scaling': 'robust' if abs(skewness) > 1 else 'standard'
            }
            
            print(f"{col}: skewness {skewness:.2f}, range {range_val:.0f}, "
                  f"scaling: {feature_analysis[col]['scaling']}")
        
        return feature_analysis
    
    def create_engineered_features(self, df):
        """Create additional features for better embeddings."""
        df_eng = df.copy()
        
        # Price-based features
        df_eng['Price_Log'] = np.log1p(df_eng['Price'])
        df_eng['Price_Rank_Category'] = df_eng.groupby('Category')['Price'].rank(pct=True)
        df_eng['Price_Zscore_Category'] = df_eng.groupby('Category')['Price'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Inventory-based features
        df_eng['Inventory_Log'] = np.log1p(df_eng['Inventory'])
        df_eng['Inventory_Rank_Category'] = df_eng.groupby('Category')['Inventory'].rank(pct=True)
        df_eng['Is_Out_Of_Stock'] = (df_eng['Inventory'] == 0).astype(int)
        df_eng['Is_Low_Stock'] = (df_eng['Inventory'] <= 5).astype(int)
        
        # Time-based features
        df_eng['Product_Age_Months'] = df_eng['Days_Since_Launch'] / 30.44
        df_eng['Is_New_Product'] = (df_eng['Days_Since_Launch'] <= 30).astype(int)
        df_eng['Is_Legacy_Product'] = (df_eng['Days_Since_Launch'] >= 365).astype(int)
        
        # Category interaction features
        df_eng['Category_Gender'] = df_eng['Category'] + '_' + df_eng['Gender']
        df_eng['Division_Category'] = df_eng['Division'] + '_' + df_eng['Category']
        
        # Color family groupings
        color_families = {
            'Neutral': ['Black', 'White', 'Gray', 'Beige', 'Brown', 'Khaki', 'Charcoal', 'Cream'],
            'Cool': ['Navy', 'Blue', 'Green', 'Purple', 'Mint', 'Lavender', 'Sage'],
            'Warm': ['Red', 'Yellow', 'Orange', 'Pink', 'Coral', 'Dusty Rose'],
            'Earth': ['Olive', 'Camel', 'Burgundy', 'Denim']
        }
        
        def get_color_family(color):
            for family, colors in color_families.items():
                if color in colors:
                    return family
            return 'Other'
        
        df_eng['Color_Family'] = df_eng['Color'].apply(get_color_family)
        
        # Size standardization (convert to numeric where possible)
        size_to_numeric = {
            'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5, 'XXL': 6,
            '0': 0, '2': 2, '4': 4, '6': 6, '8': 8, '10': 10, '12': 12, '14': 14, '16': 16,
            '28': 28, '30': 30, '32': 32, '34': 34, '36': 36, '38': 38, '40': 40, '42': 42
        }
        
        df_eng['Size_Numeric'] = df_eng['Size'].map(size_to_numeric).fillna(-1)
        df_eng['Has_Numeric_Size'] = (df_eng['Size_Numeric'] != -1).astype(int)
        
        print(f"Created {len(df_eng.columns) - len(df.columns)} engineered features")
        return df_eng
    
    def fit_transform(self, df):
        """Fit preprocessor and transform data."""
        print("Fitting preprocessing pipeline...")
        
        # Create engineered features
        df_processed = self.create_engineered_features(df)
        
        # Update feature lists to include engineered features
        new_categorical = ['Category_Gender', 'Division_Category', 'Color_Family']
        new_numerical = ['Price_Log', 'Price_Rank_Category', 'Price_Zscore_Category',
                        'Inventory_Log', 'Inventory_Rank_Category', 'Product_Age_Months',
                        'Size_Numeric', 'Is_Out_Of_Stock', 'Is_Low_Stock', 'Is_New_Product',
                        'Is_Legacy_Product', 'Has_Numeric_Size']
        
        all_categorical = self.categorical_cols + new_categorical
        all_numerical = self.numerical_cols + new_numerical
        
        # Categorical preprocessing
        categorical_preprocessor = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Numerical preprocessing
        numerical_preprocessor = Pipeline([
            ('scaler', RobustScaler())  # Using RobustScaler due to skewed data
        ])
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer([
            ('categorical', categorical_preprocessor, all_categorical),
            ('numerical', numerical_preprocessor, all_numerical)
        ])
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(df_processed)
        
        # Get feature names
        categorical_features = self.preprocessor.named_transformers_['categorical']['onehot'].get_feature_names_out(all_categorical)
        numerical_features = all_numerical
        self.feature_names = list(categorical_features) + numerical_features
        
        print(f"Processed data shape: {X_processed.shape}")
        print(f"Total features: {len(self.feature_names)}")
        
        return X_processed, self.feature_names
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df_processed = self.create_engineered_features(df)
        X_processed = self.preprocessor.transform(df_processed)
        
        return X_processed
    
    def save_preprocessor(self, filepath='fashion_preprocessor.pkl'):
        """Save fitted preprocessor for later use."""
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='fashion_preprocessor.pkl'):
        """Load saved preprocessor."""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.preprocessor = preprocessor_data['preprocessor']
        self.feature_names = preprocessor_data['feature_names']
        self.categorical_cols = preprocessor_data['categorical_cols']
        self.numerical_cols = preprocessor_data['numerical_cols']
        
        print(f"Preprocessor loaded from {filepath}")

class TargetEncodingPreprocessor:
    """
    Alternative preprocessor using target encoding for high-cardinality features.
    Useful when you have a target variable or can create one.
    """
    
    def __init__(self):
        self.target_encoders = {}
        self.numerical_scaler = None
        self.feature_names = []
    
    def create_pseudo_target(self, df):
        """Create a pseudo target variable for target encoding."""
        # Create a composite score based on price tier and inventory status
        price_scores = {'Budget': 1, 'Value': 2, 'Mid-Range': 3, 'Premium': 4, 'Luxury': 5}
        inventory_scores = {'Low Stock': 1, 'Limited': 2, 'In Stock': 3, 'Overstocked': 4}
        
        df['Price_Score'] = df['Price_Tier'].map(price_scores)
        df['Inventory_Score'] = df['Inventory_Status'].map(inventory_scores)
        
        # Composite target: weighted combination
        pseudo_target = (df['Price_Score'] * 0.6 + df['Inventory_Score'] * 0.4 + 
                        np.random.normal(0, 0.1, len(df)))  # Add noise
        
        return pseudo_target
    
    def fit_transform_target_encoding(self, df, target_col='pseudo_target'):
        """Fit and apply target encoding."""
        df_processed = df.copy()
        
        if target_col == 'pseudo_target':
            df_processed[target_col] = self.create_pseudo_target(df_processed)
        
        high_cardinality_cols = ['Subcategory', 'Size', 'Color']
        
        for col in high_cardinality_cols:
            # Calculate mean target for each category
            target_means = df_processed.groupby(col)[target_col].mean()
            
            # Apply smoothing to prevent overfitting
            global_mean = df_processed[target_col].mean()
            category_counts = df_processed[col].value_counts()
            
            # Bayesian smoothing
            smoothing_factor = 10  # Adjust based on data size
            smoothed_means = (target_means * category_counts + global_mean * smoothing_factor) / (category_counts + smoothing_factor)
            
            # Apply encoding
            df_processed[f'{col}_target_encoded'] = df_processed[col].map(smoothed_means)
            self.target_encoders[col] = smoothed_means
        
        return df_processed

def demonstrate_preprocessing():
    """Demonstrate the preprocessing pipeline."""
    print("="*60)
    print("FASHION DATA PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv('fashion_retail_dataset.csv')
    print(f"Original data shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = FashionDataPreprocessor()
    
    # Analyze features
    feature_analysis = preprocessor.analyze_features(df)
    
    # Fit and transform
    X_processed, feature_names = preprocessor.fit_transform(df)
    
    print(f"\nFinal processed shape: {X_processed.shape}")
    print(f"Feature names (first 20): {feature_names[:20]}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Demonstrate target encoding approach
    print(f"\n" + "="*40)
    print("TARGET ENCODING APPROACH")
    print("="*40)
    
    target_preprocessor = TargetEncodingPreprocessor()
    df_target_encoded = target_preprocessor.fit_transform_target_encoding(df)
    
    print(f"Target encoded features added:")
    target_encoded_cols = [col for col in df_target_encoded.columns if 'target_encoded' in col]
    for col in target_encoded_cols:
        print(f"- {col}: mean={df_target_encoded[col].mean():.3f}, std={df_target_encoded[col].std():.3f}")
    
    return X_processed, feature_names, df_target_encoded

if __name__ == "__main__":
    X_processed, feature_names, df_target_encoded = demonstrate_preprocessing()
    
    print(f"\nPreprocessing complete!")
    print(f"- Standard preprocessing: {X_processed.shape}")
    print(f"- Feature names saved")
    print(f"- Preprocessor saved for future use")
    print(f"- Alternative target encoding demonstrated")
    print(f"\nReady for embedding model training!")