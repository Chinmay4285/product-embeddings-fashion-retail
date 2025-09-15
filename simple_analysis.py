import pandas as pd
import numpy as np

def analyze_fashion_dataset():
    """Simple but comprehensive analysis of the fashion dataset."""
    print("Loading fashion retail dataset...")
    df = pd.read_csv('fashion_retail_dataset.csv')
    
    print("="*50)
    print("FASHION RETAIL DATASET ANALYSIS")
    print("="*50)
    
    # Basic overview
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Products: {len(df):,}")
    
    # Data quality
    print(f"\nData Quality:")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Duplicate rows: {df.duplicated().sum()}")
    print(f"- Unique SKUs: {df['SKU'].nunique()}/{len(df)}")
    
    # Categorical features analysis
    categorical_cols = ['Category', 'Subcategory', 'Division', 'Gender', 'Size', 
                       'Color', 'Seasonality', 'Material', 'Fit', 'Price_Tier', 
                       'Inventory_Status', 'Product_Age_Category']
    
    print(f"\nCategorical Features Cardinality:")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        most_common_pct = (df[col].value_counts().iloc[0] / len(df)) * 100
        print(f"- {col}: {unique_count} unique values (top value: {most_common_pct:.1f}%)")
    
    # Numerical features analysis
    numerical_cols = ['Price', 'Inventory', 'Days_Since_Launch']
    print(f"\nNumerical Features Statistics:")
    for col in numerical_cols:
        stats = df[col].describe()
        skewness = df[col].skew()
        print(f"- {col}: Range [{stats['min']:.1f}, {stats['max']:.1f}], Mean: {stats['mean']:.1f}, Skew: {skewness:.2f}")
    
    # Key insights for embeddings
    print(f"\nEmbedding Considerations:")
    total_onehot_dims = sum(df[col].nunique() for col in categorical_cols)
    high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
    
    print(f"- Total one-hot dimensions: {total_onehot_dims}")
    print(f"- High cardinality features (>50 values): {high_cardinality}")
    print(f"- Recommended embedding dimension: {min(50, max(10, int(np.sqrt(total_onehot_dims))))}")
    
    # Business insights
    print(f"\nBusiness Insights:")
    print(f"- Category distribution: {dict(df['Category'].value_counts().head(3))}")
    print(f"- Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
    print(f"- Average price: ${df['Price'].mean():.2f}")
    print(f"- Out of stock items: {len(df[df['Inventory'] == 0])}")
    
    # Preprocessing recommendations
    print(f"\nPreprocessing Recommendations:")
    print(f"- Use embedding layers for: {[col for col in categorical_cols if df[col].nunique() > 50]}")
    print(f"- Use one-hot encoding for: {[col for col in categorical_cols if df[col].nunique() <= 10]}")
    print(f"- Apply log transformation to: {[col for col in numerical_cols if abs(df[col].skew()) > 1]}")
    
    return df

if __name__ == "__main__":
    df = analyze_fashion_dataset()
    print(f"\nAnalysis complete! Dataset ready for embedding creation.")