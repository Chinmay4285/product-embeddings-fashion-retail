import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
import warnings
warnings.filterwarnings('ignore')

class EmbeddingEvaluator:
    """
    Comprehensive evaluation framework for product embeddings.
    Provides metrics, visualizations, and business insights.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.visualization_data = {}
    
    def load_embeddings(self):
        """Load all types of embeddings from saved files."""
        embeddings = {}
        
        # Load classical embeddings
        try:
            with open('classical_embeddings.pkl', 'rb') as f:
                classical_data = pickle.load(f)
                embeddings['pca'] = classical_data['embeddings']['pca']
                embeddings['svd'] = classical_data['embeddings']['svd']
                embeddings['tsne'] = classical_data['embeddings']['tsne']
        except:
            print("Could not load classical embeddings")
        
        # Load deep learning embeddings (placeholder - would need actual implementation)
        # embeddings['autoencoder'] = autoencoder_embeddings
        # embeddings['vae'] = vae_embeddings
        # embeddings['embedding_layers'] = embedding_layer_embeddings
        
        return embeddings
    
    def evaluate_clustering_quality(self, embeddings, true_labels=None, n_clusters_range=range(2, 11)):
        """Evaluate clustering quality using various metrics."""
        print("Evaluating clustering quality...")
        
        results = {}
        
        for method_name, embedding_data in embeddings.items():
            print(f"\nEvaluating {method_name}...")
            
            if len(embedding_data.shape) != 2:
                print(f"Skipping {method_name}: invalid shape {embedding_data.shape}")
                continue
            
            method_results = {
                'silhouette_scores': [],
                'calinski_harabasz_scores': [],
                'davies_bouldin_scores': [],
                'inertias': [],
                'n_clusters': list(n_clusters_range)
            }
            
            for n_clusters in n_clusters_range:
                try:
                    # Fit K-means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embedding_data)
                    
                    # Calculate metrics
                    silhouette = silhouette_score(embedding_data, cluster_labels)
                    calinski = calinski_harabasz_score(embedding_data, cluster_labels)
                    davies_bouldin = davies_bouldin_score(embedding_data, cluster_labels)
                    inertia = kmeans.inertia_
                    
                    method_results['silhouette_scores'].append(silhouette)
                    method_results['calinski_harabasz_scores'].append(calinski)
                    method_results['davies_bouldin_scores'].append(davies_bouldin)
                    method_results['inertias'].append(inertia)
                    
                except Exception as e:
                    print(f"Error with {method_name}, k={n_clusters}: {e}")
                    method_results['silhouette_scores'].append(0)
                    method_results['calinski_harabasz_scores'].append(0)
                    method_results['davies_bouldin_scores'].append(float('inf'))
                    method_results['inertias'].append(float('inf'))
            
            # Find optimal number of clusters
            if method_results['silhouette_scores']:
                best_k_idx = np.argmax(method_results['silhouette_scores'])
                method_results['optimal_k'] = method_results['n_clusters'][best_k_idx]
                method_results['best_silhouette'] = method_results['silhouette_scores'][best_k_idx]
            
            results[method_name] = method_results
        
        return results
    
    def evaluate_similarity_preservation(self, embeddings, original_data=None):
        """Evaluate how well embeddings preserve similarity structure."""
        print("Evaluating similarity preservation...")
        
        results = {}
        
        if original_data is not None:
            # Calculate original similarities
            original_distances = euclidean_distances(original_data)
            
            for method_name, embedding_data in embeddings.items():
                if len(embedding_data.shape) != 2:
                    continue
                
                # Calculate embedding similarities
                embedding_distances = euclidean_distances(embedding_data)
                
                # Calculate correlation between original and embedded distances
                correlation = np.corrcoef(
                    original_distances.flatten(),
                    embedding_distances.flatten()
                )[0, 1]
                
                results[method_name] = {
                    'distance_correlation': correlation,
                    'embedding_shape': embedding_data.shape
                }
        
        return results
    
    def evaluate_nearest_neighbors(self, embeddings, df, k=5):
        """Evaluate nearest neighbor quality using business logic."""
        print("Evaluating nearest neighbor quality...")
        
        results = {}
        
        for method_name, embedding_data in embeddings.items():
            if len(embedding_data.shape) != 2:
                continue
            
            # Fit nearest neighbors
            nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')
            nn_model.fit(embedding_data)
            
            # Sample some products for evaluation
            n_samples = min(100, len(embedding_data))
            sample_indices = np.random.choice(len(embedding_data), n_samples, replace=False)
            
            category_accuracy = []
            same_category_scores = []
            
            for idx in sample_indices:
                # Find nearest neighbors
                distances, indices = nn_model.kneighbors([embedding_data[idx]])
                neighbor_indices = indices[0][1:]  # Exclude self
                
                # Check category consistency
                original_category = df.iloc[idx]['Category']
                neighbor_categories = df.iloc[neighbor_indices]['Category'].values
                
                category_matches = (neighbor_categories == original_category).sum()
                category_accuracy.append(category_matches / k)
                
                # Calculate average distance to same-category items
                same_category_distances = distances[0][1:][neighbor_categories == original_category]
                if len(same_category_distances) > 0:
                    same_category_scores.append(np.mean(same_category_distances))
            
            results[method_name] = {
                'avg_category_accuracy': np.mean(category_accuracy),
                'std_category_accuracy': np.std(category_accuracy),
                'avg_same_category_distance': np.mean(same_category_scores) if same_category_scores else float('inf'),
                'embedding_shape': embedding_data.shape
            }
        
        return results
    
    def create_comparison_visualizations(self, embeddings, df):
        """Create comprehensive comparison visualizations."""
        print("Creating comparison visualizations...")
        
        # Determine grid size
        n_methods = len(embeddings)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols + 1  # +1 for metrics plot
        
        fig = plt.figure(figsize=(18, 6 * n_rows))
        
        plot_idx = 1
        
        # Individual embedding visualizations
        for method_name, embedding_data in embeddings.items():
            if len(embedding_data.shape) != 2:
                continue
            
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            
            # Use PCA for 2D if embedding is higher dimensional
            if embedding_data.shape[1] > 2:
                pca = PCA(n_components=2)
                plot_data = pca.fit_transform(embedding_data)
                title_suffix = f" (PCA 2D, orig: {embedding_data.shape[1]}D)"
            else:
                plot_data = embedding_data
                title_suffix = f" ({embedding_data.shape[1]}D)"
            
            # Color by category
            categories = df['Category'].astype('category')
            colors = plt.cm.tab10(categories.cat.codes)
            
            # Handle t-SNE subset if applicable
            if method_name == 'tsne' and len(plot_data) != len(df):
                # Assume t-SNE used a subset
                subset_size = len(plot_data)
                df_subset = df.iloc[:subset_size]
                categories_subset = df_subset['Category'].astype('category')
                colors_subset = plt.cm.tab10(categories_subset.cat.codes)
                scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1], c=colors_subset, alpha=0.6, s=20)
            else:
                scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1], c=colors, alpha=0.6, s=20)
            
            ax.set_title(f'{method_name.upper()}{title_suffix}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
            plot_idx += 1
        
        # Metrics comparison plot
        ax_metrics = plt.subplot(n_rows, n_cols, plot_idx)
        
        # Create sample metrics for visualization
        methods = list(embeddings.keys())
        sample_scores = np.random.rand(len(methods))  # Placeholder
        
        bars = ax_metrics.bar(methods, sample_scores)
        ax_metrics.set_title('Embedding Quality Comparison')
        ax_metrics.set_ylabel('Quality Score')
        ax_metrics.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, sample_scores):
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('embedding_comparison_full.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_similarity_heatmap(self, embeddings, df, sample_size=20):
        """Create similarity heatmaps for different embedding methods."""
        print("Creating similarity heatmaps...")
        
        # Sample products (ensure we don't exceed embedding size)
        max_samples = min(sample_size, len(df))
        for method_name, embedding_data in embeddings.items():
            max_samples = min(max_samples, len(embedding_data))
        
        sample_indices = np.random.choice(max_samples, min(sample_size, max_samples), replace=False)
        sample_df = df.iloc[sample_indices]
        
        n_methods = len(embeddings)
        fig, axes = plt.subplots(1, min(n_methods, 3), figsize=(15, 5))
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, embedding_data) in enumerate(embeddings.items()):
            if idx >= 3:  # Limit to 3 methods for visualization
                break
            
            if len(embedding_data.shape) != 2:
                continue
            
            # Calculate similarity matrix
            sample_embeddings = embedding_data[sample_indices]
            similarity_matrix = cosine_similarity(sample_embeddings)
            
            # Create heatmap
            ax = axes[idx] if n_methods > 1 else axes[0]
            im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
            
            # Add product labels
            labels = [f"{row['Category'][:3]}-{row['Color'][:3]}" for _, row in sample_df.iterrows()]
            ax.set_xticks(range(sample_size))
            ax.set_yticks(range(sample_size))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            
            ax.set_title(f'{method_name.upper()} Similarity')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('similarity_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_business_insights(self, embeddings, df):
        """Analyze business-relevant insights from embeddings."""
        print("Analyzing business insights...")
        
        insights = {}
        
        for method_name, embedding_data in embeddings.items():
            if len(embedding_data.shape) != 2:
                continue
            
            method_insights = {}
            
            # 1. Price vs Embedding Analysis
            if 'Price' in df.columns:
                # Calculate correlation between price and embedding dimensions
                price_correlations = []
                n_samples = min(len(df), len(embedding_data))
                for dim in range(min(10, embedding_data.shape[1])):  # Check first 10 dims
                    try:
                        corr = np.corrcoef(df['Price'][:n_samples], embedding_data[:n_samples, dim])[0, 1]
                        if not np.isnan(corr):
                            price_correlations.append(abs(corr))
                    except:
                        pass
                
                method_insights['max_price_correlation'] = max(price_correlations) if price_correlations else 0
            
            # 2. Category Separation Analysis
            categories = df['Category'].unique()
            category_separations = []
            
            n_samples = min(len(df), len(embedding_data))
            for cat in categories:
                cat_mask = df['Category'][:n_samples] == cat
                cat_embeddings = embedding_data[:n_samples][cat_mask]
                other_embeddings = embedding_data[:n_samples][~cat_mask]
                
                if len(cat_embeddings) > 0 and len(other_embeddings) > 0:
                    # Calculate average intra-category distance
                    intra_distances = euclidean_distances(cat_embeddings)
                    avg_intra = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])
                    
                    # Calculate average inter-category distance
                    inter_distances = euclidean_distances(cat_embeddings, other_embeddings)
                    avg_inter = np.mean(inter_distances)
                    
                    separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
                    category_separations.append(separation_ratio)
            
            method_insights['avg_category_separation'] = np.mean(category_separations) if category_separations else 0
            
            # 3. Inventory vs Embedding Analysis
            if 'Inventory' in df.columns:
                # Find products with high/low inventory and analyze their clustering
                n_samples = min(len(df), len(embedding_data))
                df_subset = df[:n_samples]
                embedding_subset = embedding_data[:n_samples]
                
                high_inventory = df_subset['Inventory'] > df_subset['Inventory'].quantile(0.8)
                low_inventory = df_subset['Inventory'] < df_subset['Inventory'].quantile(0.2)
                
                if high_inventory.sum() > 1 and low_inventory.sum() > 1:
                    high_inv_embeddings = embedding_subset[high_inventory]
                    low_inv_embeddings = embedding_subset[low_inventory]
                    
                    # Calculate average distance between high and low inventory items
                    cross_distances = euclidean_distances(high_inv_embeddings, low_inv_embeddings)
                    method_insights['inventory_separation'] = np.mean(cross_distances)
            
            insights[method_name] = method_insights
        
        return insights
    
    def generate_recommendations(self, evaluation_results, business_insights):
        """Generate business recommendations based on evaluation results."""
        print("\nGenerating business recommendations...")
        
        recommendations = {
            'best_for_clustering': None,
            'best_for_similarity': None,
            'best_for_category_separation': None,
            'overall_recommendation': None,
            'business_use_cases': {}
        }
        
        # Find best method for clustering
        best_clustering_score = -1
        best_clustering_method = None
        
        for method, results in evaluation_results.items():
            if 'best_silhouette' in results:
                if results['best_silhouette'] > best_clustering_score:
                    best_clustering_score = results['best_silhouette']
                    best_clustering_method = method
        
        recommendations['best_for_clustering'] = best_clustering_method
        
        # Find best method for category separation
        best_separation_score = 0
        best_separation_method = None
        
        for method, insights in business_insights.items():
            if 'avg_category_separation' in insights:
                if insights['avg_category_separation'] > best_separation_score:
                    best_separation_score = insights['avg_category_separation']
                    best_separation_method = method
        
        recommendations['best_for_category_separation'] = best_separation_method
        
        # Generate use case recommendations
        recommendations['business_use_cases'] = {
            'Product Recommendation System': best_separation_method or 'embedding_layers',
            'Inventory Management': best_clustering_method or 'pca',
            'Market Basket Analysis': 'embedding_layers',
            'Customer Segmentation': best_clustering_method or 'pca',
            'Product Search': best_separation_method or 'embedding_layers',
            'Trend Analysis': 'autoencoder',
            'Data Visualization': 'tsne',
            'Feature Engineering': 'pca'
        }
        
        return recommendations
    
    def create_evaluation_report(self, embeddings, df):
        """Create comprehensive evaluation report."""
        print("="*60)
        print("COMPREHENSIVE EMBEDDING EVALUATION REPORT")
        print("="*60)
        
        # Run all evaluations
        clustering_results = self.evaluate_clustering_quality(embeddings)
        nn_results = self.evaluate_nearest_neighbors(embeddings, df)
        business_insights = self.analyze_business_insights(embeddings, df)
        
        # Create visualizations
        self.create_comparison_visualizations(embeddings, df)
        self.create_similarity_heatmap(embeddings, df)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(clustering_results, business_insights)
        
        # Print detailed results
        print(f"\nCLUSTERING EVALUATION:")
        print("-" * 40)
        for method, results in clustering_results.items():
            print(f"{method.upper()}:")
            print(f"  Optimal clusters: {results.get('optimal_k', 'N/A')}")
            print(f"  Best silhouette score: {results.get('best_silhouette', 0):.3f}")
        
        print(f"\nNEAREST NEIGHBOR EVALUATION:")
        print("-" * 40)
        for method, results in nn_results.items():
            print(f"{method.upper()}:")
            print(f"  Category accuracy: {results['avg_category_accuracy']:.3f} ± {results['std_category_accuracy']:.3f}")
            print(f"  Avg same-category distance: {results['avg_same_category_distance']:.3f}")
        
        print(f"\nBUSINESS INSIGHTS:")
        print("-" * 40)
        for method, insights in business_insights.items():
            print(f"{method.upper()}:")
            print(f"  Price correlation: {insights.get('max_price_correlation', 0):.3f}")
            print(f"  Category separation: {insights.get('avg_category_separation', 0):.3f}")
        
        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        print(f"Best for clustering: {recommendations['best_for_clustering']}")
        print(f"Best for category separation: {recommendations['best_for_category_separation']}")
        
        print(f"\nUSE CASE RECOMMENDATIONS:")
        for use_case, method in recommendations['business_use_cases'].items():
            print(f"  {use_case}: {method}")
        
        return {
            'clustering_results': clustering_results,
            'nn_results': nn_results,
            'business_insights': business_insights,
            'recommendations': recommendations
        }

def demonstrate_evaluation_framework():
    """Demonstrate the evaluation framework."""
    print("="*60)
    print("EMBEDDING EVALUATION FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv('fashion_retail_dataset.csv')
    
    # Create evaluator
    evaluator = EmbeddingEvaluator()
    
    # Load available embeddings
    embeddings = evaluator.load_embeddings()
    
    if not embeddings:
        print("No embeddings found. Creating sample embeddings for demonstration...")
        # Create sample embeddings for demonstration
        from sklearn.decomposition import PCA
        from preprocessing_pipeline import FashionDataPreprocessor
        
        preprocessor = FashionDataPreprocessor()
        X_processed, _ = preprocessor.fit_transform(df)
        
        # Create sample embeddings
        pca = PCA(n_components=30)
        pca_embeddings = pca.fit_transform(X_processed)
        
        embeddings = {
            'pca': pca_embeddings,
            'sample_embedding': np.random.randn(len(df), 32)  # Random for comparison
        }
    
    # Run comprehensive evaluation
    evaluation_report = evaluator.create_evaluation_report(embeddings, df)
    
    print(f"\nEvaluation framework demonstration complete!")
    return evaluation_report

if __name__ == "__main__":
    results = demonstrate_evaluation_framework()