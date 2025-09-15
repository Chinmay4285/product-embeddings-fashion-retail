import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import pickle
import time
from preprocessing_pipeline import FashionDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ClassicalEmbeddingMethods:
    """
    Implementation of classical ML approaches for creating product embeddings.
    Includes PCA, t-SNE, UMAP, and evaluation methods.
    """
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca_model = None
        self.tsne_model = None
        self.umap_model = None
        self.svd_model = None
        
        self.embeddings = {}
        self.evaluation_metrics = {}
        
    def fit_pca(self, X, explained_variance_threshold=0.95):
        """
        Fit PCA and determine optimal number of components.
        """
        print("Fitting PCA...")
        
        # First, fit with all possible components to analyze explained variance
        pca_full = PCA()
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
        """
        Fit Truncated SVD (useful for sparse matrices).
        """
        print("Fitting Truncated SVD...")
        
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        svd_embeddings = self.svd_model.fit_transform(X)
        
        self.embeddings['svd'] = svd_embeddings
        self.evaluation_metrics['svd'] = {
            'explained_variance_ratio': self.svd_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.svd_model.explained_variance_ratio_),
            'n_components': self.n_components
        }
        
        print(f"SVD fitted: {svd_embeddings.shape}")
        print(f"Explained variance: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return svd_embeddings
    
    def fit_tsne(self, X, perplexity=30, n_iter=1000):
        """
        Fit t-SNE for visualization (typically 2D).
        """
        print("Fitting t-SNE...")
        
        # For t-SNE, typically use 2 or 3 dimensions for visualization
        n_tsne_components = min(3, self.n_components)
        
        # If data is high-dimensional, first reduce with PCA
        if X.shape[1] > 50:
            print("Pre-reducing dimensions with PCA for t-SNE...")
            pca_temp = PCA(n_components=50)
            X_reduced = pca_temp.fit_transform(X)
        else:
            X_reduced = X
        
        start_time = time.time()
        self.tsne_model = TSNE(
            n_components=n_tsne_components, 
            perplexity=perplexity, 
            n_iter=n_iter,
            random_state=42,
            verbose=1
        )
        tsne_embeddings = self.tsne_model.fit_transform(X_reduced)
        end_time = time.time()
        
        self.embeddings['tsne'] = tsne_embeddings
        self.evaluation_metrics['tsne'] = {
            'perplexity': perplexity,
            'n_iter': n_iter,
            'training_time': end_time - start_time,
            'n_components': n_tsne_components
        }
        
        print(f"t-SNE fitted: {tsne_embeddings.shape}")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        
        return tsne_embeddings
    
    def fit_umap(self, X, n_neighbors=15, min_dist=0.1):
        """
        Fit UMAP for both dimensionality reduction and visualization.
        """
        print("Fitting UMAP...")
        
        start_time = time.time()
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=True
        )
        umap_embeddings = self.umap_model.fit_transform(X)
        end_time = time.time()
        
        self.embeddings['umap'] = umap_embeddings
        self.evaluation_metrics['umap'] = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'training_time': end_time - start_time,
            'n_components': self.n_components
        }
        
        print(f"UMAP fitted: {umap_embeddings.shape}")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        
        return umap_embeddings
    
    def evaluate_embeddings(self, X_original, df_original):
        """
        Evaluate embedding quality using various metrics.
        """
        print("\nEvaluating embedding quality...")
        
        evaluation_results = {}
        
        for method_name, embeddings in self.embeddings.items():
            print(f"\nEvaluating {method_name.upper()}...")
            
            # Clustering evaluation
            if embeddings.shape[1] >= 2:  # Need at least 2D for clustering
                # Try different number of clusters
                silhouette_scores = []
                k_range = range(2, min(11, len(np.unique(df_original['Category'])) + 3))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, cluster_labels)
                    silhouette_scores.append(score)
                
                best_k = k_range[np.argmax(silhouette_scores)]
                best_silhouette = max(silhouette_scores)
                
                # Final clustering with best k
                kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                cluster_labels = kmeans_best.fit_predict(embeddings)
                
                evaluation_results[method_name] = {
                    'best_k_clusters': best_k,
                    'best_silhouette_score': best_silhouette,
                    'silhouette_scores': silhouette_scores,
                    'cluster_labels': cluster_labels,
                    'embedding_shape': embeddings.shape
                }
                
                print(f"Best clusters: {best_k}, Silhouette score: {best_silhouette:.3f}")
                
                # Category purity analysis
                if 'Category' in df_original.columns:
                    category_purity = self.calculate_category_purity(
                        cluster_labels, df_original['Category']
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
            cluster_categories = true_categories[cluster_mask]
            
            if len(cluster_categories) > 0:
                most_common_category = cluster_categories.value_counts().iloc[0]
                purity = most_common_category / len(cluster_categories)
                purity_scores.append(purity)
        
        return np.mean(purity_scores)
    
    def create_visualizations(self, df_original):
        """Create comprehensive visualizations of embeddings."""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fashion Product Embeddings Comparison', fontsize=16)
        
        plot_idx = 0
        
        for method_name, embeddings in self.embeddings.items():
            if embeddings.shape[1] >= 2:  # Can plot 2D
                ax = axes[plot_idx // 3, plot_idx % 3]
                
                # Color by category
                categories = df_original['Category'].astype('category')
                scatter = ax.scatter(
                    embeddings[:, 0], 
                    embeddings[:, 1],
                    c=categories.cat.codes,
                    cmap='tab10',
                    alpha=0.6,
                    s=20
                )
                
                ax.set_title(f'{method_name.upper()} Embedding (by Category)')
                ax.set_xlabel(f'{method_name.upper()} Component 1')
                ax.set_ylabel(f'{method_name.upper()} Component 2')
                
                # Add legend for categories
                unique_categories = categories.unique()
                for i, cat in enumerate(unique_categories):
                    ax.scatter([], [], c=plt.cm.tab10(i), label=cat)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plot_idx += 1
        
        # PCA explained variance plot
        if 'pca' in self.evaluation_metrics:
            ax = axes[1, 2]
            cumvar = self.evaluation_metrics['pca']['cumulative_variance']
            ax.plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
            ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Explained Variance')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('classical_embeddings_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'classical_embeddings_visualization.png'")
    
    def save_models(self, filepath_prefix='classical_embeddings'):
        """Save all fitted models and embeddings."""
        models_data = {
            'pca_model': self.pca_model,
            'tsne_model': self.tsne_model,
            'umap_model': self.umap_model,
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
        self.fit_umap(X)
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
    print("RECOMMENDATIONS FOR GAP'S USE CASE")
    print("="*40)
    
    print(f"\n1. PCA:")
    print(f"   - Best for: Initial dimensionality reduction, feature analysis")
    print(f"   - Pros: Fast, interpretable, linear relationships")
    print(f"   - Cons: May miss non-linear patterns")
    
    print(f"\n2. UMAP:")
    print(f"   - Best for: General-purpose embedding, preserves local/global structure")
    print(f"   - Pros: Fast, good for clustering, preserves meaning")
    print(f"   - Cons: Hyperparameter sensitive")
    
    print(f"\n3. t-SNE:")
    print(f"   - Best for: Visualization, exploration")
    print(f"   - Pros: Great for 2D/3D visualization")
    print(f"   - Cons: Slow, not suitable for new data")
    
    print(f"\n4. SVD:")
    print(f"   - Best for: Large sparse datasets, alternative to PCA")
    print(f"   - Pros: Memory efficient, handles sparse data")
    print(f"   - Cons: Similar limitations to PCA")
    
    return embeddings, evaluation_results

if __name__ == "__main__":
    embeddings, results = demonstrate_classical_embeddings()
    
    print(f"\nClassical embedding methods demonstration complete!")
    print(f"All models, embeddings, and visualizations saved.")
    print(f"Ready to proceed with deep learning approaches!")