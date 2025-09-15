import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FashionDataAnalyzer:
    """
    Comprehensive data analysis and quality assessment for fashion retail data.
    """
    
    def __init__(self, data_path='fashion_retail_dataset.csv'):
        """Initialize with data loading and basic setup."""
        self.df = pd.read_csv(data_path)
        self.original_shape = self.df.shape
        self.categorical_cols = ['Category', 'Subcategory', 'Division', 'Gender', 'Size', 
                                'Color', 'Seasonality', 'Material', 'Fit', 'Price_Tier', 
                                'Inventory_Status', 'Product_Age_Category']
        self.numerical_cols = ['Price', 'Inventory', 'Days_Since_Launch']
        
    def data_overview(self):
        """Provide comprehensive data overview."""
        print("="*50)
        print("FASHION RETAIL DATASET OVERVIEW")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Products: {len(self.df):,}")
        print(f"Total Features: {len(self.df.columns)}")
        
        print(f"\nFeature Types:")
        print(f"- Categorical: {len(self.categorical_cols)}")
        print(f"- Numerical: {len(self.numerical_cols)}")
        print(f"- Identifiers: {len(['SKU', 'Product_Name', 'Launch_Date'])}")
        
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df.info()
    
    def data_quality_check(self):
        """Comprehensive data quality assessment."""
        print("\n" + "="*50)
        print("DATA QUALITY ASSESSMENT")
        print("="*50)
        
        # Missing values
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        print("\nMissing Values:")
        if missing_data.sum() == 0:
            print("* No missing values found")
        else:
            missing_df = pd.DataFrame({
                'Count': missing_data[missing_data > 0],
                'Percentage': missing_pct[missing_pct > 0]
            })
            print(missing_df)
        
        # Duplicate records
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate Records: {duplicates}")
        if duplicates == 0:
            print("* No duplicate records found")
        
        # SKU uniqueness
        sku_unique = self.df['SKU'].nunique()
        print(f"\nSKU Uniqueness: {sku_unique}/{len(self.df)}")
        if sku_unique == len(self.df):
            print("* All SKUs are unique")
        
        # Data type consistency
        print(f"\nData Types:")
        for col in self.df.columns:
            print(f"- {col}: {self.df[col].dtype}")
        
        return {
            'missing_values': missing_data.sum(),
            'duplicates': duplicates,
            'sku_unique': sku_unique == len(self.df)
        }
    
    def categorical_analysis(self):
        """Analyze categorical features for embedding considerations."""
        print("\n" + "="*50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*50)
        
        categorical_stats = {}
        
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            most_common = self.df[col].value_counts().head(3)
            
            print(f"\n{col}:")
            print(f"  Unique values: {unique_count}")
            print(f"  Top 3 values:")
            for val, count in most_common.items():
                pct = (count/len(self.df))*100
                print(f"    {val}: {count} ({pct:.1f}%)")
            
            categorical_stats[col] = {
                'unique_count': unique_count,
                'cardinality': 'high' if unique_count > 50 else 'medium' if unique_count > 10 else 'low',
                'distribution': 'uniform' if most_common.iloc[0] / len(self.df) < 0.3 else 'skewed'
            }
        
        return categorical_stats
    
    def numerical_analysis(self):
        """Analyze numerical features."""
        print("\n" + "="*50)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*50)
        
        for col in self.numerical_cols:
            print(f"\n{col}:")
            stats = self.df[col].describe()
            print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Skewness: {self.df[col].skew():.2f}")
            
            # Check for outliers using IQR method
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)][col]
            print(f"  Outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
        
        return self.df[self.numerical_cols].describe()
    
    def create_visualizations(self):
        """Create comprehensive visualizations for data understanding."""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Category distribution
        plt.subplot(4, 3, 1)
        self.df['Category'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Product Distribution by Category')
        plt.xticks(rotation=45)
        
        # 2. Price distribution
        plt.subplot(4, 3, 2)
        plt.hist(self.df['Price'], bins=50, color='lightgreen', alpha=0.7)
        plt.title('Price Distribution')
        plt.xlabel('Price ($)')
        
        # 3. Division breakdown
        plt.subplot(4, 3, 3)
        self.df['Division'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Division Distribution')
        
        # 4. Color popularity
        plt.subplot(4, 3, 4)
        top_colors = self.df['Color'].value_counts().head(10)
        top_colors.plot(kind='barh', color='coral')
        plt.title('Top 10 Colors')
        
        # 5. Price by category
        plt.subplot(4, 3, 5)
        self.df.boxplot(column='Price', by='Category', ax=plt.gca())
        plt.title('Price Distribution by Category')
        plt.xticks(rotation=45)
        
        # 6. Inventory status
        plt.subplot(4, 3, 6)
        self.df['Inventory_Status'].value_counts().plot(kind='bar', color='orange')
        plt.title('Inventory Status Distribution')
        plt.xticks(rotation=45)
        
        # 7. Size distribution
        plt.subplot(4, 3, 7)
        size_counts = self.df['Size'].value_counts().head(15)
        size_counts.plot(kind='bar', color='purple')
        plt.title('Top 15 Sizes')
        plt.xticks(rotation=45)
        
        # 8. Price tier distribution
        plt.subplot(4, 3, 8)
        self.df['Price_Tier'].value_counts().plot(kind='bar', color='brown')
        plt.title('Price Tier Distribution')
        plt.xticks(rotation=45)
        
        # 9. Material distribution
        plt.subplot(4, 3, 9)
        self.df['Material'].value_counts().plot(kind='bar', color='pink')
        plt.title('Material Distribution')
        plt.xticks(rotation=45)
        
        # 10. Correlation heatmap for numerical features
        plt.subplot(4, 3, 10)
        corr_matrix = self.df[self.numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Numerical Features Correlation')
        
        # 11. Launch date trend
        plt.subplot(4, 3, 11)
        self.df['Launch_Date'] = pd.to_datetime(self.df['Launch_Date'])
        launch_counts = self.df.groupby(self.df['Launch_Date'].dt.to_period('M')).size()
        launch_counts.plot(kind='line', color='red')
        plt.title('Product Launches Over Time')
        plt.xticks(rotation=45)
        
        # 12. Gender vs Category heatmap
        plt.subplot(4, 3, 12)
        gender_category = pd.crosstab(self.df['Gender'], self.df['Category'])
        sns.heatmap(gender_category, annot=True, fmt='d', cmap='Blues')
        plt.title('Gender vs Category Distribution')
        
        plt.tight_layout()
        plt.savefig('fashion_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("* Visualizations saved as 'fashion_data_analysis.png'")
    
    def preprocessing_requirements(self):
        """Identify preprocessing requirements for embedding creation."""
        print("\n" + "="*50)
        print("PREPROCESSING REQUIREMENTS")
        print("="*50)
        
        requirements = {
            'categorical_encoding': {},
            'numerical_scaling': {},
            'feature_engineering': [],
            'embedding_considerations': {}
        }
        
        # Categorical encoding requirements
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count > 50:
                requirements['categorical_encoding'][col] = 'embedding_layer'
                print(f"* {col}: Use embedding layer (high cardinality: {unique_count})")
            elif unique_count > 10:
                requirements['categorical_encoding'][col] = 'target_encoding'
                print(f"* {col}: Consider target encoding (medium cardinality: {unique_count})")
            else:
                requirements['categorical_encoding'][col] = 'one_hot'
                print(f"* {col}: Use one-hot encoding (low cardinality: {unique_count})")
        
        print(f"\nNumerical scaling requirements:")
        for col in self.numerical_cols:
            stats = self.df[col].describe()
            range_val = stats['max'] - stats['min']
            skewness = self.df[col].skew()
            
            if abs(skewness) > 1:
                requirements['numerical_scaling'][col] = 'log_transform_then_standard'
                print(f"* {col}: Log transform + StandardScaler (highly skewed: {skewness:.2f})")
            elif range_val > 1000:
                requirements['numerical_scaling'][col] = 'robust_scaler'
                print(f"* {col}: RobustScaler (large range: {range_val:.0f})")
            else:
                requirements['numerical_scaling'][col] = 'standard_scaler'
                print(f"* {col}: StandardScaler")
        
        # Feature engineering opportunities
        print(f"\nFeature engineering opportunities:")
        opportunities = [
            "Create price percentile within category",
            "Calculate inventory turnover ratio",
            "Generate seasonal preference scores",
            "Create size standardization mapping",
            "Build color family groupings",
            "Generate product lifecycle stage features"
        ]
        
        for opp in opportunities:
            print(f"* {opp}")
            requirements['feature_engineering'].append(opp)
        
        # Embedding specific considerations
        print(f"\nEmbedding considerations:")
        total_categorical_dims = sum(self.df[col].nunique() for col in self.categorical_cols)
        print(f"* Total categorical dimensions if one-hot: {total_categorical_dims}")
        print(f"* Recommended embedding dimension: {min(50, max(10, int(np.sqrt(total_categorical_dims))))}")
        print(f"* High cardinality features for embedding layers: {[col for col in self.categorical_cols if self.df[col].nunique() > 50]}")
        
        requirements['embedding_considerations'] = {
            'total_onehot_dims': total_categorical_dims,
            'recommended_embed_dim': min(50, max(10, int(np.sqrt(total_categorical_dims)))),
            'high_cardinality_features': [col for col in self.categorical_cols if self.df[col].nunique() > 50]
        }
        
        return requirements
    
    def run_full_analysis(self):
        """Run complete data analysis pipeline."""
        self.data_overview()
        quality_results = self.data_quality_check()
        categorical_stats = self.categorical_analysis()
        numerical_stats = self.numerical_analysis()
        self.create_visualizations()
        preprocessing_reqs = self.preprocessing_requirements()
        
        return {
            'quality_results': quality_results,
            'categorical_stats': categorical_stats,
            'numerical_stats': numerical_stats,
            'preprocessing_requirements': preprocessing_reqs
        }

if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = FashionDataAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("* Data quality assessment completed")
    print("* Feature analysis completed")
    print("* Visualizations generated")
    print("* Preprocessing requirements identified")
    print("\nReady for embedding implementation!")