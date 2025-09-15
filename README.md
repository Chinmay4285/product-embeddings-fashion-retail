
# GAP Product Embeddings - Production Files

## Files Created:
- classical_pca_embeddings.npy: Classical PCA embeddings ((1000, 32))
- autoencoder_embeddings.npy: Autoencoder embeddings ((1000, 32))
- entity_embeddings.npy: Entity embeddings ((1000, 32))
- autoencoder_model.h5: Trained autoencoder model
- entity_model.h5: Trained entity embedding model
- preprocessor.pkl: Data preprocessor
- product_metadata.csv: Product information

## Usage:
1. Load embeddings: np.load('autoencoder_embeddings.npy')
2. Load model: keras.models.load_model('autoencoder_model.h5')
3. Load preprocessor: pickle.load(open('preprocessor.pkl', 'rb'))
4. Load metadata: pd.read_csv('product_metadata.csv')

## Best Performing Method: Entity Embeddings
