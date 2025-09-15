import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class EmbeddingLayerNetwork:
    """
    Neural network that learns embeddings for categorical features.
    Useful for high-cardinality categorical features.
    """
    
    def __init__(self, embedding_dims=None, dense_dims=[128, 64], output_dim=32):
        self.embedding_dims = embedding_dims or {}
        self.dense_dims = dense_dims
        self.output_dim = output_dim
        
        self.model = None
        self.embedding_model = None
        self.label_encoders = {}
        self.feature_info = {}
        self.history = None
        
    def prepare_categorical_data(self, df):
        """Prepare categorical features for embedding layers."""
        print("Preparing categorical data for embedding layers...")
        
        categorical_cols = ['Category', 'Subcategory', 'Division', 'Gender', 'Size', 
                           'Color', 'Seasonality', 'Material', 'Fit']
        
        # Encode categorical variables
        encoded_data = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df[col].astype(str))
                encoded_data[col] = encoded_values
                
                self.label_encoders[col] = le
                vocab_size = len(le.classes_)
                
                # Calculate embedding dimension (rule of thumb: min(50, vocab_size//2))
                embed_dim = min(50, max(4, vocab_size // 2))
                if col in self.embedding_dims:
                    embed_dim = self.embedding_dims[col]
                
                self.feature_info[col] = {
                    'vocab_size': vocab_size,
                    'embedding_dim': embed_dim,
                    'classes': le.classes_
                }
                
                print(f"{col}: vocab_size={vocab_size}, embedding_dim={embed_dim}")
        
        return encoded_data
    
    def prepare_numerical_data(self, df):
        """Prepare numerical features."""
        numerical_cols = ['Price', 'Inventory', 'Days_Since_Launch']
        
        numerical_data = {}
        for col in numerical_cols:
            if col in df.columns:
                # Normalize numerical features
                values = df[col].values.astype(np.float32)
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
                numerical_data[col] = normalized
        
        return numerical_data
    
    def create_embedding_model(self, encoded_data, numerical_data):
        """Create model with embedding layers for categorical features."""
        print("Building neural network with embedding layers...")
        
        # Input layers for categorical features
        categorical_inputs = {}
        embedding_layers = {}
        
        for col, vocab_info in self.feature_info.items():
            input_layer = keras.Input(shape=(1,), name=f'{col}_input')
            embedding_layer = layers.Embedding(
                input_dim=vocab_info['vocab_size'],
                output_dim=vocab_info['embedding_dim'],
                name=f'{col}_embedding'
            )(input_layer)
            embedding_layer = layers.Flatten()(embedding_layer)
            
            categorical_inputs[col] = input_layer
            embedding_layers[col] = embedding_layer
        
        # Input layer for numerical features
        numerical_input = keras.Input(shape=(len(numerical_data),), name='numerical_input')
        
        # Concatenate all embeddings and numerical features
        all_features = list(embedding_layers.values()) + [numerical_input]
        concat_layer = layers.Concatenate()(all_features)
        
        # Dense layers
        x = concat_layer
        for dim in self.dense_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
        
        # Final embedding output
        embedding_output = layers.Dense(self.output_dim, activation='relu', name='final_embedding')(x)
        
        # Create model
        all_inputs = list(categorical_inputs.values()) + [numerical_input]
        self.embedding_model = keras.Model(inputs=all_inputs, outputs=embedding_output)
        
        print("Embedding model architecture:")
        self.embedding_model.summary()
        
        return self.embedding_model
    
    def create_classification_model(self, encoded_data, numerical_data, target_col='Category'):
        """Create a classification model to learn meaningful embeddings."""
        print(f"Building classification model with target: {target_col}")
        
        # Create embedding model
        embedding_model = self.create_embedding_model(encoded_data, numerical_data)
        
        # Add classification head
        embeddings = embedding_model.output
        
        # Classification layers
        x = layers.Dense(64, activation='relu')(embeddings)
        x = layers.Dropout(0.3)(x)
        
        # Output layer for classification
        num_classes = len(self.label_encoders[target_col].classes_)
        classification_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
        
        # Full model
        self.model = keras.Model(inputs=embedding_model.inputs, outputs=classification_output)
        
        # Compile
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Classification model architecture:")
        self.model.summary()
        
        return self.model
    
    def prepare_training_data(self, encoded_data, numerical_data, df, target_col='Category'):
        """Prepare data for training."""
        # Categorical inputs (with correct naming)
        categorical_inputs = {}
        for col in self.feature_info.keys():
            categorical_inputs[f'{col}_input'] = encoded_data[col].reshape(-1, 1)
        
        # Numerical input
        numerical_input = np.column_stack(list(numerical_data.values()))
        
        # Combine inputs
        X = categorical_inputs
        X['numerical_input'] = numerical_input
        
        # Target
        y = self.label_encoders[target_col].transform(df[target_col].astype(str))
        
        return X, y
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model."""
        print("Training embedding model...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def get_embeddings(self, X):
        """Extract embeddings from the trained model."""
        if self.embedding_model is None:
            raise ValueError("Model not trained yet!")
        
        embeddings = self.embedding_model.predict(X)
        return embeddings
    
    def get_categorical_embeddings(self):
        """Extract learned embeddings for each categorical feature."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        categorical_embeddings = {}
        
        for col, vocab_info in self.feature_info.items():
            embedding_layer = self.model.get_layer(f'{col}_embedding')
            embeddings = embedding_layer.get_weights()[0]  # Shape: (vocab_size, embedding_dim)
            
            categorical_embeddings[col] = {
                'embeddings': embeddings,
                'classes': vocab_info['classes'],
                'vocab_size': vocab_info['vocab_size'],
                'embedding_dim': vocab_info['embedding_dim']
            }
        
        return categorical_embeddings
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('embedding_network_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_embeddings(self, feature_name='Color', max_items=20):
        """Visualize learned embeddings for a specific categorical feature."""
        categorical_embeddings = self.get_categorical_embeddings()
        
        if feature_name not in categorical_embeddings:
            print(f"Feature {feature_name} not found in embeddings")
            return
        
        embeddings = categorical_embeddings[feature_name]['embeddings']
        classes = categorical_embeddings[feature_name]['classes']
        
        # Use first 2 dimensions or PCA if higher dimensional
        if embeddings.shape[1] >= 2:
            embed_2d = embeddings[:, :2]
        else:
            print(f"Embeddings for {feature_name} are 1D, cannot visualize")
            return
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Show only top items to avoid overcrowding
        n_items = min(max_items, len(classes))
        for i in range(n_items):
            plt.scatter(embed_2d[i, 0], embed_2d[i, 1], s=100, alpha=0.7)
            plt.annotate(classes[i], (embed_2d[i, 0], embed_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'Learned Embeddings for {feature_name}')
        plt.xlabel('Embedding Dimension 1')
        plt.ylabel('Embedding Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{feature_name.lower()}_embeddings_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath_prefix='embedding_network'):
        """Save the trained models and encoders."""
        # Save models
        self.model.save(f"{filepath_prefix}_full.h5")
        self.embedding_model.save(f"{filepath_prefix}_embeddings.h5")
        
        # Save configuration
        config = {
            'embedding_dims': self.embedding_dims,
            'dense_dims': self.dense_dims,
            'output_dim': self.output_dim,
            'label_encoders': self.label_encoders,
            'feature_info': self.feature_info,
            'history': self.history.history if self.history else None
        }
        
        with open(f"{filepath_prefix}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Models and configuration saved with prefix: {filepath_prefix}")

def demonstrate_embedding_layers():
    """Demonstrate neural network with embedding layers."""
    print("="*60)
    print("NEURAL NETWORK WITH EMBEDDING LAYERS DEMONSTRATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv('fashion_retail_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Initialize embedding network
    embedding_network = EmbeddingLayerNetwork(
        embedding_dims={'Color': 8, 'Size': 10, 'Subcategory': 12},  # Custom dimensions
        dense_dims=[128, 64],
        output_dim=32
    )
    
    # Prepare data
    encoded_data = embedding_network.prepare_categorical_data(df)
    numerical_data = embedding_network.prepare_numerical_data(df)
    
    # Create classification model (learns embeddings via category prediction)
    model = embedding_network.create_classification_model(
        encoded_data, numerical_data, target_col='Category'
    )
    
    # Prepare training data
    X, y = embedding_network.prepare_training_data(
        encoded_data, numerical_data, df, target_col='Category'
    )
    
    print(f"Training data prepared:")
    print(f"- Categorical inputs: {len([k for k in X.keys() if k != 'numerical_input'])}")
    print(f"- Numerical features: {X['numerical_input'].shape[1]}")
    print(f"- Target classes: {len(np.unique(y))}")
    
    # Train model
    history = embedding_network.train(X, y, epochs=30, batch_size=64)
    
    # Plot training history
    embedding_network.plot_training_history()
    
    # Get final embeddings for all products
    product_embeddings = embedding_network.get_embeddings(X)
    print(f"Product embeddings shape: {product_embeddings.shape}")
    
    # Get categorical embeddings
    categorical_embeddings = embedding_network.get_categorical_embeddings()
    
    # Visualize embeddings for different features
    for feature in ['Color', 'Material', 'Category']:
        if feature in categorical_embeddings:
            embedding_network.visualize_embeddings(feature, max_items=15)
    
    # Save model
    embedding_network.save_model('fashion_embedding_network')
    
    # COMPARISON WITH DIFFERENT APPROACHES
    print("\n" + "="*50)
    print("EMBEDDING APPROACHES COMPARISON")
    print("="*50)
    
    # Compare embedding dimensions and interpretability
    print("\nLearned Categorical Embeddings:")
    for feature, info in categorical_embeddings.items():
        embeddings = info['embeddings']
        print(f"\n{feature}:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Sample classes: {info['classes'][:5]}...")
        
        # Find similar items (using cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Show most similar pairs
        n_classes = len(info['classes'])
        for i in range(min(3, n_classes)):
            sim_scores = similarities[i]
            most_similar_idx = np.argsort(sim_scores)[-2]  # -1 is self
            print(f"  {info['classes'][i]} most similar to: {info['classes'][most_similar_idx]} "
                  f"(similarity: {sim_scores[most_similar_idx]:.3f})")
    
    # BUSINESS INSIGHTS
    print("\n" + "="*50)
    print("BUSINESS INSIGHTS")
    print("="*50)
    
    print("\n1. Embedding Layer Approach Benefits:")
    print("   - Learns dense representations for high-cardinality categorical features")
    print("   - Captures semantic relationships between categories")
    print("   - Handles new categories better than one-hot encoding")
    print("   - More memory efficient than one-hot for large vocabularies")
    
    print("\n2. Use Cases for GAP:")
    print("   - Product similarity search using learned embeddings")
    print("   - Recommendation systems based on product relationships")
    print("   - Market basket analysis with product embeddings")
    print("   - Customer segmentation using purchase embedding patterns")
    
    print("\n3. Production Considerations:")
    print("   - Can handle streaming data with new product attributes")
    print("   - Embeddings can be pre-computed and stored for fast retrieval")
    print("   - Model can be retrained periodically to capture new trends")
    print("   - Embeddings transfer well to related downstream tasks")
    
    return {
        'model': embedding_network,
        'product_embeddings': product_embeddings,
        'categorical_embeddings': categorical_embeddings,
        'feature_info': embedding_network.feature_info
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    results = demonstrate_embedding_layers()
    
    print(f"\nEmbedding layers demonstration complete!")
    print(f"Learned embeddings for categorical features.")
    print(f"Models and visualizations saved.")
    print(f"Ready for evaluation framework!")