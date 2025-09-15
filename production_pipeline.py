import pandas as pd
import numpy as np
import pickle
import h5py
import sqlite3
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductEmbeddingPipeline:
    """
    Production-ready pipeline for creating, storing, and retrieving product embeddings.
    Supports batch processing, incremental updates, and fast similarity search.
    """
    
    def __init__(self, storage_path: str = "embeddings_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize storage components
        self.db_path = self.storage_path / "embeddings.db"
        self.embeddings_file = self.storage_path / "embeddings.h5"
        self.models_path = self.storage_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Loaded components
        self.preprocessor = None
        self.embedding_models = {}
        self.similarity_index = None
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Products table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT UNIQUE NOT NULL,
                    category TEXT,
                    subcategory TEXT,
                    division TEXT,
                    gender TEXT,
                    size TEXT,
                    color TEXT,
                    price REAL,
                    inventory INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Embeddings metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    embedding_method TEXT NOT NULL,
                    embedding_dimension INTEGER,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sku) REFERENCES products (sku)
                )
            """)
            
            # Similarity cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS similarity_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_sku TEXT NOT NULL,
                    target_sku TEXT NOT NULL,
                    similarity_score REAL,
                    method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index separately
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_method 
                ON similarity_cache (source_sku, method)
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def store_products(self, df: pd.DataFrame):
        """Store product data in the database."""
        logger.info(f"Storing {len(df)} products in database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data for insertion
            product_data = []
            for _, row in df.iterrows():
                product_data.append((
                    row['SKU'],
                    row.get('Category', ''),
                    row.get('Subcategory', ''),
                    row.get('Division', ''),
                    row.get('Gender', ''),
                    row.get('Size', ''),
                    row.get('Color', ''),
                    row.get('Price', 0.0),
                    row.get('Inventory', 0)
                ))
            
            # Insert with conflict resolution
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO products 
                (sku, category, subcategory, division, gender, size, color, price, inventory)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, product_data)
            
            conn.commit()
            logger.info(f"Successfully stored {cursor.rowcount} products")
    
    def store_embeddings(self, embeddings: Dict[str, np.ndarray], method: str, 
                        model_version: str = "1.0", skus: Optional[List[str]] = None):
        """Store embeddings in HDF5 format for efficient retrieval."""
        logger.info(f"Storing embeddings using method: {method}")
        
        # Store embeddings in HDF5
        with h5py.File(self.embeddings_file, 'a') as f:
            group_name = f"{method}_{model_version.replace('.', '_')}"
            
            # Remove existing group if it exists
            if group_name in f:
                del f[group_name]
            
            group = f.create_group(group_name)
            
            if isinstance(embeddings, dict):
                # Multiple embedding types
                for embed_type, embed_data in embeddings.items():
                    group.create_dataset(embed_type, data=embed_data)
            else:
                # Single embedding array
                group.create_dataset('embeddings', data=embeddings)
            
            # Store metadata
            group.attrs['method'] = method
            group.attrs['model_version'] = model_version
            group.attrs['created_at'] = datetime.now().isoformat()
            group.attrs['dimension'] = embeddings.shape[1] if hasattr(embeddings, 'shape') else 0
        
        # Store metadata in database
        if skus is not None:
            self._store_embedding_metadata(skus, method, embeddings.shape[1], model_version)
        
        logger.info(f"Successfully stored embeddings for method: {method}")
    
    def _store_embedding_metadata(self, skus: List[str], method: str, 
                                 dimension: int, model_version: str):
        """Store embedding metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing metadata for this method
            cursor.execute("""
                DELETE FROM embeddings_metadata 
                WHERE embedding_method = ? AND model_version = ?
            """, (method, model_version))
            
            # Insert new metadata
            metadata = [(sku, method, dimension, model_version) for sku in skus]
            cursor.executemany("""
                INSERT INTO embeddings_metadata 
                (sku, embedding_method, embedding_dimension, model_version)
                VALUES (?, ?, ?, ?)
            """, metadata)
            
            conn.commit()
    
    def load_embeddings(self, method: str, model_version: str = "1.0") -> Optional[np.ndarray]:
        """Load embeddings from storage."""
        group_name = f"{method}_{model_version.replace('.', '_')}"
        
        try:
            with h5py.File(self.embeddings_file, 'r') as f:
                if group_name not in f:
                    logger.warning(f"Embeddings not found for {method} v{model_version}")
                    return None
                
                group = f[group_name]
                
                # Load main embeddings dataset
                if 'embeddings' in group:
                    embeddings = group['embeddings'][:]
                else:
                    # Load first available dataset
                    dataset_name = list(group.keys())[0]
                    embeddings = group[dataset_name][:]
                
                logger.info(f"Loaded embeddings: {embeddings.shape} for {method}")
                return embeddings
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def build_similarity_index(self, embeddings: np.ndarray, method: str = 'cosine'):
        """Build fast similarity search index."""
        from sklearn.neighbors import NearestNeighbors
        
        logger.info(f"Building similarity index with {method} metric...")
        
        metric = 'cosine' if method == 'cosine' else 'euclidean'
        self.similarity_index = NearestNeighbors(
            n_neighbors=100,  # Top 100 similar items
            metric=metric,
            algorithm='ball_tree' if metric == 'euclidean' else 'brute'
        )
        
        self.similarity_index.fit(embeddings)
        logger.info("Similarity index built successfully")
    
    def find_similar_products(self, product_sku: str, n_similar: int = 10, 
                            method: str = 'cosine') -> List[Tuple[str, float]]:
        """Find similar products using the similarity index."""
        if self.similarity_index is None:
            logger.error("Similarity index not built. Call build_similarity_index first.")
            return []
        
        # Get product index
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM products WHERE sku = ?", (product_sku,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Product {product_sku} not found")
                return []
            
            product_idx = result[0] - 1  # Convert to 0-based index
        
        # Find similar products
        try:
            embeddings = self.load_embeddings('pca')  # Default to PCA
            if embeddings is None:
                return []
            
            distances, indices = self.similarity_index.kneighbors(
                [embeddings[product_idx]], n_neighbors=n_similar + 1
            )
            
            # Get SKUs for similar products
            similar_skus = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):
                    cursor.execute("SELECT sku FROM products WHERE rowid = ?", (idx + 1,))
                    result = cursor.fetchone()
                    if result:
                        similarity_score = 1 - dist if method == 'cosine' else 1 / (1 + dist)
                        similar_skus.append((result[0], similarity_score))
            
            return similar_skus
            
        except Exception as e:
            logger.error(f"Error finding similar products: {e}")
            return []
    
    def batch_process_new_products(self, new_products_df: pd.DataFrame):
        """Process new products through the complete pipeline."""
        logger.info(f"Processing {len(new_products_df)} new products...")
        
        # Store products
        self.store_products(new_products_df)
        
        # Load preprocessor
        if self.preprocessor is None:
            try:
                from preprocessing_pipeline import FashionDataPreprocessor
                self.preprocessor = FashionDataPreprocessor()
                self.preprocessor.load_preprocessor()
            except:
                logger.error("Could not load preprocessor")
                return
        
        # Generate embeddings
        try:
            X_processed, _ = self.preprocessor.fit_transform(new_products_df)
            
            # Apply PCA (assuming it's already fitted)
            from sklearn.decomposition import PCA
            import pickle
            
            # Load fitted PCA model
            with open('classical_embeddings.pkl', 'rb') as f:
                classical_models = pickle.load(f)
                pca_model = classical_models['pca_model']
            
            if pca_model:
                embeddings = pca_model.transform(X_processed)
                skus = new_products_df['SKU'].tolist()
                
                self.store_embeddings(embeddings, 'pca', '1.0', skus)
                
                # Update similarity index
                all_embeddings = self.load_embeddings('pca')
                if all_embeddings is not None:
                    self.build_similarity_index(all_embeddings)
                
                logger.info("Successfully processed new products")
            
        except Exception as e:
            logger.error(f"Error processing new products: {e}")
    
    def get_product_recommendations(self, product_sku: str, n_recommendations: int = 5) -> Dict:
        """Get comprehensive product recommendations."""
        similar_products = self.find_similar_products(product_sku, n_recommendations)
        
        if not similar_products:
            return {'error': 'No recommendations found'}
        
        # Get detailed product information
        recommendations = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for sku, similarity in similar_products:
                cursor.execute("""
                    SELECT sku, category, subcategory, color, price, inventory
                    FROM products WHERE sku = ?
                """, (sku,))
                
                result = cursor.fetchone()
                if result:
                    recommendations.append({
                        'sku': result[0],
                        'category': result[1],
                        'subcategory': result[2],
                        'color': result[3],
                        'price': result[4],
                        'inventory': result[5],
                        'similarity_score': similarity
                    })
        
        return {
            'source_product': product_sku,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    
    def export_embeddings(self, method: str, output_format: str = 'csv') -> str:
        """Export embeddings to various formats."""
        embeddings = self.load_embeddings(method)
        if embeddings is None:
            return ""
        
        output_file = self.storage_path / f"{method}_embeddings.{output_format}"
        
        if output_format == 'csv':
            # Get SKUs
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT sku FROM products ORDER BY rowid")
                skus = [row[0] for row in cursor.fetchall()]
            
            # Create DataFrame
            columns = [f'dim_{i}' for i in range(embeddings.shape[1])]
            df = pd.DataFrame(embeddings, columns=columns)
            df.insert(0, 'sku', skus[:len(embeddings)])
            df.to_csv(output_file, index=False)
            
        elif output_format == 'npy':
            np.save(output_file, embeddings)
            
        elif output_format == 'parquet':
            # Similar to CSV but in Parquet format
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT sku FROM products ORDER BY rowid")
                skus = [row[0] for row in cursor.fetchall()]
            
            columns = [f'dim_{i}' for i in range(embeddings.shape[1])]
            df = pd.DataFrame(embeddings, columns=columns)
            df.insert(0, 'sku', skus[:len(embeddings)])
            df.to_parquet(output_file, index=False)
        
        logger.info(f"Exported embeddings to {output_file}")
        return str(output_file)
    
    def get_analytics(self) -> Dict:
        """Get analytics about the stored embeddings."""
        analytics = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Product counts
            cursor.execute("SELECT COUNT(*) FROM products")
            analytics['total_products'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT category, COUNT(*) FROM products GROUP BY category")
            analytics['products_by_category'] = dict(cursor.fetchall())
            
            # Embedding methods
            cursor.execute("""
                SELECT embedding_method, model_version, COUNT(*) 
                FROM embeddings_metadata 
                GROUP BY embedding_method, model_version
            """)
            analytics['embedding_methods'] = {}
            for method, version, count in cursor.fetchall():
                analytics['embedding_methods'][f"{method}_v{version}"] = count
        
        # Storage size
        analytics['storage_size_mb'] = {
            'database': self.db_path.stat().st_size / (1024*1024),
            'embeddings': self.embeddings_file.stat().st_size / (1024*1024) if self.embeddings_file.exists() else 0
        }
        
        return analytics

def demonstrate_production_pipeline():
    """Demonstrate the production pipeline."""
    print("="*60)
    print("PRODUCTION PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ProductEmbeddingPipeline("production_embeddings")
    
    # Load sample data
    df = pd.read_csv('fashion_retail_dataset.csv')
    print(f"Loaded {len(df)} products")
    
    # Store products
    pipeline.store_products(df)
    
    # Load and store embeddings (using PCA from previous examples)
    try:
        with open('classical_embeddings.pkl', 'rb') as f:
            classical_data = pickle.load(f)
            embeddings = classical_data['embeddings']['pca']
            
        pipeline.store_embeddings(embeddings, 'pca', '1.0', df['SKU'].tolist())
        print("Stored PCA embeddings")
        
        # Build similarity index
        pipeline.build_similarity_index(embeddings)
        
        # Test recommendations
        sample_sku = df['SKU'].iloc[0]
        recommendations = pipeline.get_product_recommendations(sample_sku, 5)
        
        print(f"\nRecommendations for {sample_sku}:")
        if 'recommendations' in recommendations:
            for rec in recommendations['recommendations']:
                print(f"  {rec['sku']}: {rec['category']} - {rec['color']} (similarity: {rec['similarity_score']:.3f})")
        else:
            print(f"  Error: {recommendations.get('error', 'Unknown error')}")
        
        # Export embeddings
        export_file = pipeline.export_embeddings('pca', 'csv')
        print(f"\nExported embeddings to: {export_file}")
        
        # Get analytics
        analytics = pipeline.get_analytics()
        print(f"\nPipeline Analytics:")
        print(f"  Total products: {analytics['total_products']}")
        print(f"  Storage size: {analytics['storage_size_mb']['database']:.1f}MB (DB) + {analytics['storage_size_mb']['embeddings']:.1f}MB (embeddings)")
        print(f"  Embedding methods: {list(analytics['embedding_methods'].keys())}")
        
    except Exception as e:
        print(f"Error in pipeline demonstration: {e}")
        print("Note: This requires embeddings from previous steps")
    
    print(f"\nProduction pipeline setup complete!")
    print(f"Storage location: {pipeline.storage_path}")
    
    return pipeline

if __name__ == "__main__":
    pipeline = demonstrate_production_pipeline()