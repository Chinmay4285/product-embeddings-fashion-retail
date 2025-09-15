import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import time
from preprocessing_pipeline import FashionDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ClassicalEmbeddingMethods:
    """
    Implementation of classical ML approaches for creating product embeddings.
    Includes PCA, t-SNE, and evaluation methods.
    """
    
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.pca_model = None
        self.tsne_model = None
        self.svd_model = None
        
        self.embeddings = {}
        self.evaluation_metrics = {}
        
    def fit_pca(self, X, explained_variance_threshold=0.95):
        """Fit PCA and determine optimal number of components."""
        print("Fitting PCA...")
        
        # First, fit with all possible components to analyze explained variance
        max_components = min(X.shape[0], X.shape[1])
        pca_full = PCA(n_components=max_components)
        pca_full.fit(X)
        
        # Find number of components for desired explained variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_optimal = np.argmax(cumsum_var >= explained_variance_threshold) + 1
        
        print(f"Components for {explained_variance_threshold*100}% variance: {n_components_optimal}")
        print(f"Using {min(self.n_components, n_components_optimal)} components")
        
        # Fit final PCA model
        n_final = min(self.n_components, n_components_optimal)
        self.pca_model = PCA(n_components=n_final)
        pca_embeddings = self.pca_model.fit_transform(X)
        
        # Store results
        self.embeddings['pca'] = pca_embeddings
        self.evaluation_metrics['pca'] = {
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca_model.explained_variance_ratio_),
            'n_components': n_final
        }
        
        print(f"PCA fitted: {pca_embeddings.shape}")
        print(f"Explained variance: {self.pca_model.explained_variance_ratio_.sum():.3f}")
        
        return pca_embeddings
    
    def fit_truncated_svd(self, X):
        """Fit Truncated SVD (useful for sparse matrices)."""
        print("Fitting Truncated SVD...")
        
        n_components = min(self.n_components, min(X.shape) - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        svd_embeddings = self.svd_model.fit_transform(X)
        
        self.embeddings['svd'] = svd_embeddings
        self.evaluation_metrics['svd'] = {
            'explained_variance_ratio': self.svd_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.svd_model.explained_variance_ratio_),
            'n_components': n_components
        }
        
        print(f"SVD fitted: {svd_embeddings.shape}")
        print(f"Explained variance: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return svd_embeddings
    
    def fit_tsne(self, X, perplexity=30, n_iter=300):
        """Fit t-SNE for visualization (typically 2D)."""
        print("Fitting t-SNE...")
        
        # For t-SNE, use 2D for visualization
        n_tsne_components = 2
        
        # If data is high-dimensional, first reduce with PCA
        if X.shape[1] > 50:
            print("Pre-reducing dimensions with PCA for t-SNE...")
            pca_temp = PCA(n_components=50)
            X_reduced = pca_temp.fit_transform(X)
        else:
            X_reduced = X
        
        # Use a subset for faster computation if dataset is large
        if X_reduced.shape[0] > 1000:
            print("Using subset of 1000 samples for t-SNE...")
            indices = np.random.choice(X_reduced.shape[0], 1000, replace=False)
            X_tsne = X_reduced[indices]
        else:
            X_tsne = X_reduced
            indices = np.arange(X_reduced.shape[0])
        
        start_time = time.time()
        self.tsne_model = TSNE(
            n_components=n_tsne_components, 
            perplexity=min(perplexity, len(X_tsne)//4), 
            n_iter=n_iter,
            random_state=42
        )
        tsne_embeddings = self.tsne_model.fit_transform(X_tsne)
        end_time = time.time()
        
        # Store results with indices for subset
        self.embeddings['tsne'] = tsne_embeddings
        self.embeddings['tsne_indices'] = indices
        self.evaluation_metrics['tsne'] = {
            'perplexity': perplexity,
            'n_iter': n_iter,
            'training_time': end_time - start_time,
            'n_components': n_tsne_components,
            'n_samples_used': len(X_tsne)
        }
        
        print(f"t-SNE fitted: {tsne_embeddings.shape}")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        
        return tsne_embeddings
    
    def evaluate_embeddings(self, X_original, df_original):
        """Evaluate embedding quality using various metrics."""
        print("\nEvaluating embedding quality...")
        
        evaluation_results = {}
        
        for method_name, embeddings in self.embeddings.items():
            if method_name.endswith('_indices'):  # Skip index arrays
                continue
                
            print(f"\nEvaluating {method_name.upper()}...")
            
            # Get corresponding subset of original data if needed
            if method_name == 'tsne' and 'tsne_indices' in self.embeddings:
                df_subset = df_original.iloc[self.embeddings['tsne_indices']]
            else:
                df_subset = df_original
            
            # Clustering evaluation
            if embeddings.shape[1] >= 2:  # Need at least 2D for clustering
                # Try different number of clusters
                silhouette_scores = []
                k_range = range(2, min(11, len(np.unique(df_subset['Category'])) + 3))
                
                best_score = -1
                best_k = 2
                
                for k in k_range:
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(embeddings)
                        score = silhouette_score(embeddings, cluster_labels)
                        silhouette_scores.append(score)
                        
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except:
                        silhouette_scores.append(-1)
                
                # Final clustering with best k
                kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                cluster_labels = kmeans_best.fit_predict(embeddings)
                
                evaluation_results[method_name] = {
                    'best_k_clusters': best_k,
                    'best_silhouette_score': best_score,
                    'silhouette_scores': silhouette_scores,
                    'cluster_labels': cluster_labels,
                    'embedding_shape': embeddings.shape
                }
                
                print(f"Best clusters: {best_k}, Silhouette score: {best_score:.3f}")
                
                # Category purity analysis
                if 'Category' in df_subset.columns:
                    category_purity = self.calculate_category_purity(
                        cluster_labels, df_subset['Category']
                    )
                    evaluation_results[method_name]['category_purity'] = category_purity
                    print(f"Category purity: {category_purity:.3f}")
        
        self.evaluation_metrics.update(evaluation_results)
        return evaluation_results
    
    def calculate_category_purity(self, cluster_labels, true_categories):
        """Calculate how well clusters align with true categories."""
        purity_scores = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_categories = true_categories.iloc[cluster_mask] if hasattr(true_categories, 'iloc') else true_categories[cluster_mask]
            
            if len(cluster_categories) > 0:
                if hasattr(cluster_categories, 'value_counts'):
                    most_common_category = cluster_categories.value_counts().iloc[0]
                else:
                    from collections import Counter
                    most_common_category = Counter(cluster_categories).most_common(1)[0][1]
                purity = most_common_category / len(cluster_categories)
                purity_scores.append(purity)
        
        return np.mean(purity_scores) if purity_scores else 0.0
    
    def create_visualizations(self, df_original):
        """Create comprehensive visualizations of embeddings."""
        print("\nCreating visualizations...")
        
        # Count available 2D+ embeddings
        available_embeddings = [(name, emb) for name, emb in self.embeddings.items() 
                               if not name.endswith('_indices') and emb.shape[1] >= 2]
        
        n_plots = len(available_embeddings) + 1  # +1 for PCA variance plot
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Fashion Product Embeddings Comparison', fontsize=16)
        
        plot_idx = 0
        
        for method_name, embeddings in available_embeddings:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = axes[row, col]
            
            # Get corresponding data subset
            if method_name == 'tsne' and 'tsne_indices' in self.embeddings:
                df_subset = df_original.iloc[self.embeddings['tsne_indices']]
            else:
                df_subset = df_original
            
            # Color by category
            categories = df_subset['Category'].astype('category')
            unique_cats = categories.unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
            
            for i, cat in enumerate(unique_cats):
                mask = categories == cat
                ax.scatter(
                    embeddings[mask, 0], 
                    embeddings[mask, 1],
                    c=[colors[i]],
                    label=cat,
                    alpha=0.6,
                    s=20
                )
            
            ax.set_title(f'{method_name.upper()} Embedding (by Category)')
            ax.set_xlabel(f'{method_name.upper()} Component 1')
            ax.set_ylabel(f'{method_name.upper()} Component 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plot_idx += 1
        
        # PCA explained variance plot
        if 'pca' in self.evaluation_metrics and 'cumulative_variance' in self.evaluation_metrics['pca']:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = axes[row, col]
            cumvar = self.evaluation_metrics['pca']['cumulative_variance']
            ax.plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
            ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Explained Variance')
            ax.legend()
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(plot_idx + 1, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('classical_embeddings_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'classical_embeddings_visualization.png'")
    
    def save_models(self, filepath_prefix='classical_embeddings'):
        """Save all fitted models and embeddings."""
        models_data = {
            'pca_model': self.pca_model,
            'tsne_model': self.tsne_model,
            'svd_model': self.svd_model,
            'embeddings': self.embeddings,
            'evaluation_metrics': self.evaluation_metrics,
            'n_components': self.n_components
        }
        
        filepath = f"{filepath_prefix}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(models_data, f)
        
        print(f"Models and embeddings saved to {filepath}")
    
    def fit_all_methods(self, X, df_original):
        """Fit all classical embedding methods."""
        print("="*60)
        print("FITTING ALL CLASSICAL EMBEDDING METHODS")
        print("="*60)
        
        # Fit all methods
        self.fit_pca(X)
        self.fit_truncated_svd(X)
        self.fit_tsne(X)  # Do this last as it's slowest
        
        # Evaluate all methods
        evaluation_results = self.evaluate_embeddings(X, df_original)
        
        # Create visualizations
        self.create_visualizations(df_original)
        
        # Save everything
        self.save_models()
        
        return self.embeddings, evaluation_results

def demonstrate_classical_embeddings():
    """Demonstrate classical embedding methods on fashion data."""
    print("="*60)
    print("CLASSICAL EMBEDDING METHODS DEMONSTRATION")
    print("="*60)
    
    # Load and preprocess data
    df = pd.read_csv('fashion_retail_dataset.csv')
    preprocessor = FashionDataPreprocessor()
    X_processed, feature_names = preprocessor.fit_transform(df)
    
    print(f"Data shape: {X_processed.shape}")
    
    # Initialize embedding methods
    embedding_methods = ClassicalEmbeddingMethods(n_components=30)
    
    # Fit all methods
    embeddings, evaluation_results = embedding_methods.fit_all_methods(X_processed, df)
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING METHODS COMPARISON")
    print("="*60)
    
    for method, results in evaluation_results.items():
        print(f"\n{method.upper()}:")
        print(f"  Embedding shape: {results['embedding_shape']}")
        print(f"  Best silhouette score: {results['best_silhouette_score']:.3f}")
        print(f"  Best number of clusters: {results['best_k_clusters']}")
        if 'category_purity' in results:
            print(f"  Category purity: {results['category_purity']:.3f}")
    
    # Method recommendations
    print(f"\n" + "="*40)
    print("RECOMMENDATIONS FOR FashionCore'S USE CASE")
    print("="*40)
    
    print(f"\n1. PCA:")
    print(f"   - Best for: Initial dimensionality reduction, feature analysis")
    print(f"   - Pros: Fast, interpretable, linear relationships")
    print(f"   - Cons: May miss non-linear patterns")
    print(f"   - Use when: You need fast, explainable dimensionality reduction")
    
    print(f"\n2. t-SNE:")
    print(f"   - Best for: Data exploration and visualization")
    print(f"   - Pros: Great for 2D/3D visualization, reveals clusters")
    print(f"   - Cons: Slow, not suitable for new data, only for visualization")
    print(f"   - Use when: You want to visualize product relationships")
    
    print(f"\n3. SVD:")
    print(f"   - Best for: Large sparse datasets, alternative to PCA")
    print(f"   - Pros: Memory efficient, handles sparse data well")
    print(f"   - Cons: Similar limitations to PCA")
    print(f"   - Use when: Memory is a constraint or data is sparse")
    
    return embeddings, evaluation_results

if __name__ == "__main__":
    embeddings, results = demonstrate_classical_embeddings()
    
    print(f"\nClassical embedding methods demonstration complete!")
    print(f"All models, embeddings, and visualizations saved.")
    print(f"Ready to proceed with deep learning approaches!")