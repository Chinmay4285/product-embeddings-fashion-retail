import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocessing_pipeline import FashionDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class AutoencoderEmbedding:
    """
    Deep learning autoencoder for creating product embeddings.
    """
    
    def __init__(self, encoding_dim=32, hidden_dims=[128, 64]):
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        
    def build_autoencoder(self, input_dim):
        """Build the autoencoder architecture."""
        print(f"Building autoencoder: {input_dim} -> {self.encoding_dim}")
        
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for hidden_dim in self.hidden_dims:
            encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Bottleneck (embedding layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='embedding')(encoded)
        
        # Decoder
        decoded = encoded
        for hidden_dim in reversed(self.hidden_dims):
            decoded = layers.Dense(hidden_dim, activation='relu')(decoded)
            decoded = layers.Dropout(0.2)(decoded)
        
        # Output layer
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Create models
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        # Compile
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Autoencoder architecture:")
        self.autoencoder.summary()
        
        return self.autoencoder
    
    def train(self, X_train, X_val, epochs=100, batch_size=32):
        """Train the autoencoder."""
        print("Training autoencoder...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
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
        self.history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def get_embeddings(self, X):
        """Extract embeddings using the trained encoder."""
        if self.encoder is None:
            raise ValueError("Model not trained yet!")
        
        embeddings = self.encoder.predict(X)
        return embeddings
    
    def evaluate_reconstruction(self, X_test):
        """Evaluate reconstruction quality."""
        if self.autoencoder is None:
            raise ValueError("Model not trained yet!")
        
        X_reconstructed = self.autoencoder.predict(X_test)
        mse = mean_squared_error(X_test.flatten(), X_reconstructed.flatten())
        
        print(f"Reconstruction MSE: {mse:.6f}")
        return mse, X_reconstructed
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('autoencoder_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath_prefix='autoencoder_embedding'):
        """Save the trained models."""
        self.autoencoder.save(f"{filepath_prefix}_full.h5")
        self.encoder.save(f"{filepath_prefix}_encoder.h5")
        
        # Save configuration
        config = {
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'history': self.history.history if self.history else None
        }
        
        with open(f"{filepath_prefix}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Models saved with prefix: {filepath_prefix}")

class VariationalAutoencoder:
    """
    Variational Autoencoder for learning probabilistic embeddings.
    """
    
    def __init__(self, encoding_dim=32, hidden_dims=[128, 64]):
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.history = None
        
    def sampling(self, args):
        """Reparameterization trick."""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def build_vae(self, input_dim):
        """Build the VAE architecture."""
        print(f"Building VAE: {input_dim} -> {self.encoding_dim}")
        
        # Encoder
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = encoder_inputs
        
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        z_mean = layers.Dense(self.encoding_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.encoding_dim, name='z_log_var')(x)
        z = layers.Lambda(self.sampling, output_shape=(self.encoding_dim,), name='z')([z_mean, z_log_var])
        
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.encoding_dim,))
        x = latent_inputs
        
        for hidden_dim in reversed(self.hidden_dims):
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = keras.Model(encoder_inputs, outputs, name='vae')
        
        # VAE loss
        reconstruction_loss = keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        
        print("VAE architecture:")
        self.vae.summary()
        
        return self.vae
    
    def train(self, X_train, X_val, epochs=100, batch_size=32):
        """Train the VAE."""
        print("Training VAE...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Train
        self.history = self.vae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("VAE training completed!")
        return self.history
    
    def get_embeddings(self, X):
        """Extract embeddings using the trained encoder."""
        if self.encoder is None:
            raise ValueError("Model not trained yet!")
        
        z_mean, _, _ = self.encoder.predict(X)
        return z_mean

def demonstrate_deep_learning_embeddings():
    """Demonstrate deep learning embedding approaches."""
    print("="*60)
    print("DEEP LEARNING EMBEDDINGS DEMONSTRATION")
    print("="*60)
    
    # Load and preprocess data
    df = pd.read_csv('fashion_retail_dataset.csv')
    preprocessor = FashionDataPreprocessor()
    X_processed, feature_names = preprocessor.fit_transform(df)
    
    # Normalize data for neural networks
    X_normalized = (X_processed - X_processed.min(axis=0)) / (X_processed.max(axis=0) - X_processed.min(axis=0) + 1e-8)
    
    # Split data
    X_train, X_temp = train_test_split(X_normalized, test_size=0.3, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # APPROACH 1: Standard Autoencoder
    print("\n" + "="*40)
    print("APPROACH 1: STANDARD AUTOENCODER")
    print("="*40)
    
    autoencoder = AutoencoderEmbedding(encoding_dim=32, hidden_dims=[128, 64])
    autoencoder.build_autoencoder(X_train.shape[1])
    
    # Train autoencoder
    autoencoder.train(X_train, X_val, epochs=50, batch_size=64)
    
    # Get embeddings
    train_embeddings_ae = autoencoder.get_embeddings(X_train)
    test_embeddings_ae = autoencoder.get_embeddings(X_test)
    
    # Evaluate reconstruction
    mse_ae, X_reconstructed_ae = autoencoder.evaluate_reconstruction(X_test)
    
    # Plot training history
    autoencoder.plot_training_history()
    
    # Save model
    autoencoder.save_model('fashion_autoencoder')
    
    # APPROACH 2: Variational Autoencoder
    print("\n" + "="*40)
    print("APPROACH 2: VARIATIONAL AUTOENCODER")
    print("="*40)
    
    vae = VariationalAutoencoder(encoding_dim=32, hidden_dims=[128, 64])
    vae.build_vae(X_train.shape[1])
    
    # Train VAE
    vae.train(X_train, X_val, epochs=50, batch_size=64)
    
    # Get embeddings
    train_embeddings_vae = vae.get_embeddings(X_train)
    test_embeddings_vae = vae.get_embeddings(X_test)
    
    print(f"VAE embeddings shape: {test_embeddings_vae.shape}")
    
    # COMPARISON AND VISUALIZATION
    print("\n" + "="*40)
    print("EMBEDDING COMPARISON")
    print("="*40)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Deep Learning Embeddings Comparison', fontsize=16)
    
    # Get subset of data for visualization
    n_vis = min(500, len(X_test))
    vis_indices = np.random.choice(len(X_test), n_vis, replace=False)
    test_df_subset = df.iloc[-len(X_test):].iloc[vis_indices]
    
    # Autoencoder embeddings (2D projection)
    from sklearn.decomposition import PCA
    pca_ae = PCA(n_components=2)
    ae_2d = pca_ae.fit_transform(test_embeddings_ae[vis_indices])
    
    categories = test_df_subset['Category'].astype('category')
    colors = plt.cm.tab10(categories.cat.codes)
    
    axes[0, 0].scatter(ae_2d[:, 0], ae_2d[:, 1], c=colors, alpha=0.6)
    axes[0, 0].set_title('Autoencoder Embeddings (PCA 2D)')
    
    # VAE embeddings (2D projection)
    pca_vae = PCA(n_components=2)
    vae_2d = pca_vae.fit_transform(test_embeddings_vae[vis_indices])
    
    axes[0, 1].scatter(vae_2d[:, 0], vae_2d[:, 1], c=colors, alpha=0.6)
    axes[0, 1].set_title('VAE Embeddings (PCA 2D)')
    
    # Reconstruction comparison
    original_sample = X_test[0:1]
    ae_reconstruction = autoencoder.autoencoder.predict(original_sample)
    
    axes[1, 0].plot(original_sample.flatten()[:50], label='Original', alpha=0.7)
    axes[1, 0].plot(ae_reconstruction.flatten()[:50], label='Reconstructed', alpha=0.7)
    axes[1, 0].set_title('Autoencoder Reconstruction (first 50 features)')
    axes[1, 0].legend()
    
    # Embedding distribution comparison
    axes[1, 1].hist(test_embeddings_ae.flatten(), alpha=0.5, bins=50, label='Autoencoder', density=True)
    axes[1, 1].hist(test_embeddings_vae.flatten(), alpha=0.5, bins=50, label='VAE', density=True)
    axes[1, 1].set_title('Embedding Value Distributions')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('deep_learning_embeddings_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # PERFORMANCE METRICS
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    
    print(f"Autoencoder:")
    print(f"  Reconstruction MSE: {mse_ae:.6f}")
    print(f"  Embedding dimension: {test_embeddings_ae.shape[1]}")
    print(f"  Compression ratio: {X_train.shape[1]/test_embeddings_ae.shape[1]:.1f}x")
    
    print(f"\nVAE:")
    print(f"  Embedding dimension: {test_embeddings_vae.shape[1]}")
    print(f"  Compression ratio: {X_train.shape[1]/test_embeddings_vae.shape[1]:.1f}x")
    
    # BUSINESS RECOMMENDATIONS
    print("\n" + "="*40)
    print("BUSINESS RECOMMENDATIONS")
    print("="*40)
    
    print("1. Standard Autoencoder:")
    print("   - Best for: General-purpose embeddings with good reconstruction")
    print("   - Pros: Deterministic, interpretable, good for similarity search")
    print("   - Cons: May overfit, no uncertainty quantification")
    print("   - Use case: Product similarity, recommendation systems")
    
    print("\n2. Variational Autoencoder:")
    print("   - Best for: Probabilistic embeddings, generation")
    print("   - Pros: Regularized, can generate new products, uncertainty")
    print("   - Cons: More complex, harder to train")
    print("   - Use case: Product generation, uncertainty-aware recommendations")
    
    return {
        'autoencoder': autoencoder,
        'vae': vae,
        'ae_embeddings': test_embeddings_ae,
        'vae_embeddings': test_embeddings_vae,
        'test_data': X_test,
        'test_df': test_df_subset
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    results = demonstrate_deep_learning_embeddings()
    
    print(f"\nDeep learning embeddings demonstration complete!")
    print(f"Models and visualizations saved.")
    print(f"Ready for embedding layer approach!")