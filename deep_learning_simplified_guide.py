"""
DEEP LEARNING FOR PRODUCT EMBEDDINGS: BEGINNER'S GUIDE
======================================================

A step-by-step explanation of deep learning for embeddings,
designed for someone completely new to the field.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

def explain_embeddings_basics():
    """
    Chapter 1: What are Embeddings? (The Foundation)
    """
    
    print("="*60)
    print("CHAPTER 1: WHAT ARE EMBEDDINGS?")
    print("="*60)
    
    print("""
SIMPLE ANALOGY:
Think of embeddings like a "smart GPS coordinate system" for products.

Instead of just saying "Red T-shirt" and "Blue T-shirt", embeddings give 
each product a set of numbers that describe its characteristics:

Traditional way:
- Red T-shirt = "Red T-shirt"
- Blue T-shirt = "Blue T-shirt"
Computer sees: Two completely different text strings

Embedding way:
- Red T-shirt = [0.8, 0.2, 0.9, 0.1] 
- Blue T-shirt = [0.1, 0.2, 0.9, 0.1]
Computer sees: Similar items! (3 out of 4 numbers are close)

WHAT DO THE NUMBERS MEAN?
Each position might represent something like:
- Position 1: How "bright" the color is (0.8 vs 0.1)
- Position 2: How "casual" it is (both 0.2 = both casual)
- Position 3: How "cotton-like" it is (both 0.9 = both cotton)
- Position 4: How "expensive" it is (both 0.1 = both cheap)

WHY THIS IS POWERFUL:
1. Computer can find similar products automatically
2. Can recommend items with compatible characteristics
3. Can group products by similarity
4. Works with thousands of products efficiently
""")

def demonstrate_simple_example():
    """
    Chapter 2: Simple Example
    """
    
    print("="*60)
    print("CHAPTER 2: SIMPLE EXAMPLE")
    print("="*60)
    
    print("\nLet's create embeddings for 5 colors:")
    
    # Simple example with colors
    colors = ["Red", "Pink", "Blue", "Navy", "Green"]
    
    # Manual embeddings (what we want to learn automatically)
    color_embeddings = {
        "Red":   [0.9, 0.8, 0.1],  # bright, warm, not-blue
        "Pink":  [0.8, 0.9, 0.1],  # bright, warm, not-blue  
        "Blue":  [0.7, 0.2, 0.9],  # bright, cool, blue
        "Navy":  [0.2, 0.2, 0.8],  # dark, cool, blue
        "Green": [0.6, 0.4, 0.3]   # medium, neutral, not-blue
    }
    
    print("Color embeddings (3 dimensions):")
    print("Format: [brightness, warmth, blueness]")
    print()
    
    for color, embedding in color_embeddings.items():
        print(f"{color:5}: {embedding}")
    
    print("\nNow let's see which colors are similar:")
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    
    embeddings_array = np.array(list(color_embeddings.values()))
    similarities = cosine_similarity(embeddings_array)
    
    print("\nSimilarity scores (higher = more similar):")
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors):
            if i < j:  # Only show each pair once
                similarity = similarities[i][j]
                print(f"{color1} & {color2}: {similarity:.3f}")
    
    print("\nNotice:")
    print("- Red & Pink are very similar (both warm, bright)")
    print("- Blue & Navy are similar (both blue-ish)")  
    print("- Red & Navy are different (warm vs cool)")

def explain_neural_networks():
    """
    Chapter 3: How Neural Networks Learn Embeddings
    """
    
    print("="*60)
    print("CHAPTER 3: HOW NEURAL NETWORKS LEARN")
    print("="*60)
    
    print("""
THE LEARNING PROCESS:

STEP 1: Start with random numbers
Red = [0.1, 0.5, 0.9] (random)
Blue = [0.7, 0.2, 0.4] (random)

STEP 2: Try to predict something (like product category)
Computer: "Red product... I think it's a Jacket!" (wrong)
Human: "No, it's a T-shirt."

STEP 3: Adjust the numbers to be better
Red = [0.2, 0.4, 0.8] (adjusted)
Computer: "Red product... I think it's a T-shirt!" (correct!)

STEP 4: Repeat thousands of times
After training, Red's numbers contain learned knowledge about red products!

WHY THIS WORKS:
- Similar products get used in similar contexts
- The computer learns to give similar numbers to similar products
- The final numbers (embeddings) capture meaningful relationships

NEURAL NETWORK STRUCTURE:

Input Features → Hidden Layers → Output Prediction
     ↓              ↓              ↓
[Color, Size]  [Learning]     [Category]
[Red, Large]   [Math Magic]   [T-shirt]

The "Hidden Layers" learn useful representations (embeddings)!
""")

def build_simple_neural_network():
    """
    Chapter 4: Building Your First Neural Network
    """
    
    print("="*60)
    print("CHAPTER 4: BUILDING YOUR FIRST NEURAL NETWORK")
    print("="*60)
    
    print("\nSTEP 1: Create sample data")
    
    # Create simple product data
    products = [
        [1, 1, 0],  # Red, Large, Cheap -> T-shirt
        [1, 0, 0],  # Red, Small, Cheap -> T-shirt
        [0, 1, 1],  # Blue, Large, Expensive -> Jacket  
        [0, 0, 1],  # Blue, Small, Expensive -> Jacket
        [1, 1, 1],  # Red, Large, Expensive -> Jacket
        [0, 0, 0],  # Blue, Small, Cheap -> T-shirt
    ]
    
    # Categories: 0 = T-shirt, 1 = Jacket
    categories = [0, 0, 1, 1, 1, 0]
    
    print("Our training data:")
    print("Format: [is_red, is_large, is_expensive] -> category")
    
    for i, (prod, cat) in enumerate(zip(products, categories)):
        category_name = "T-shirt" if cat == 0 else "Jacket"
        print(f"Product {i+1}: {prod} -> {category_name}")
    
    # Convert to numpy arrays
    X = np.array(products, dtype=float)
    y = np.array(categories)
    
    print(f"\nData shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    print("\nSTEP 2: Build the neural network")
    
    # Build simple network
    model = keras.Sequential([
        keras.Input(shape=(3,)),  # 3 input features
        layers.Dense(4, activation='relu', name='hidden'),  # Hidden layer with 4 neurons
        layers.Dense(1, activation='sigmoid', name='output')  # Output layer
    ])
    
    print("Network structure:")
    print("Input (3) -> Hidden (4) -> Output (1)")
    print("\nWhat each layer does:")
    print("- Input: Receives product features")
    print("- Hidden: Learns patterns and creates embeddings")  
    print("- Output: Predicts T-shirt (0) or Jacket (1)")
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nSTEP 3: Train the network")
    
    # Train
    print("Training in progress...")
    history = model.fit(X, y, epochs=100, verbose=0)
    
    final_accuracy = history.history['accuracy'][-1]
    print(f"Final accuracy: {final_accuracy:.3f}")
    
    print("\nSTEP 4: Test the trained model")
    
    # Test on original data
    predictions = model.predict(X, verbose=0)
    
    print("\nPredictions on training data:")
    for i, (prod, true_cat, pred) in enumerate(zip(products, categories, predictions)):
        pred_cat = "T-shirt" if pred[0] < 0.5 else "Jacket"
        true_cat_name = "T-shirt" if true_cat == 0 else "Jacket"
        confidence = pred[0] if pred[0] > 0.5 else (1 - pred[0])
        
        correct = "✓" if pred_cat == true_cat_name else "✗"
        print(f"Product {prod} -> Predicted: {pred_cat} (conf: {confidence:.3f}) {correct}")
    
    print("\nSTEP 5: Extract learned embeddings")
    
    # Get hidden layer activations (these are embeddings!)
    hidden_model = keras.Model(inputs=model.input, 
                              outputs=model.get_layer('hidden').output)
    embeddings = hidden_model.predict(X, verbose=0)
    
    print("\nLearned embeddings from hidden layer:")
    print("These 4 numbers now represent each product!")
    
    for i, (prod, emb) in enumerate(zip(products, embeddings)):
        print(f"Product {prod} -> Embedding: [{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, {emb[3]:.3f}]")
    
    return model, embeddings

def explain_embedding_layers():
    """
    Chapter 5: Embedding Layers for Categories
    """
    
    print("="*60)
    print("CHAPTER 5: EMBEDDING LAYERS FOR CATEGORIES")
    print("="*60)
    
    print("""
THE CATEGORY PROBLEM:

You have categories like colors: ["Red", "Blue", "Green", "Pink"]

OPTION 1 - Simple numbers:
Red=1, Blue=2, Green=3, Pink=4
PROBLEM: Computer thinks Blue(2) is closer to Red(1) than Pink(4)
But actually Red and Pink are more similar colors!

OPTION 2 - One-hot encoding:
Red = [1,0,0,0], Blue = [0,1,0,0], Green = [0,0,1,0], Pink = [0,0,0,1]  
PROBLEM: All colors look equally different
PROBLEM: Takes lots of memory (1000 colors = 1000 dimensions!)

OPTION 3 - Embedding layers (BEST!):
Let the computer learn dense representations!

HOW EMBEDDING LAYERS WORK:

1. START: Each category gets random numbers
   Red = [0.1, 0.8, 0.3]
   Blue = [0.9, 0.2, 0.7] 
   Pink = [0.4, 0.9, 0.1]

2. TRAINING: Computer learns to predict categories
   "Products with Red are often Casual..."
   Computer adjusts Red's numbers to be good at predicting "Casual"

3. RESULT: Similar categories get similar numbers!
   Red = [0.8, 0.1, 0.9]   (bright, casual, warm)
   Pink = [0.7, 0.2, 0.8]  (bright, casual, warm) <- Similar to Red!
   Navy = [0.2, 0.9, 0.1]  (dark, formal, cool) <- Different!

BENEFITS:
- Automatically discovers relationships  
- Much smaller than one-hot (25 colors -> 8 dimensions vs 25)
- Captures semantic meaning
- Works with new categories
""")

def demonstrate_embedding_layer():
    """
    Chapter 6: Embedding Layer Demo
    """
    
    print("="*60)
    print("CHAPTER 6: EMBEDDING LAYER DEMONSTRATION")
    print("="*60)
    
    print("\nSTEP 1: Create category data")
    
    # Create example data with colors and product types
    data = [
        ("Red", "Casual"),
        ("Red", "Casual"),
        ("Pink", "Casual"),
        ("Pink", "Casual"), 
        ("Navy", "Formal"),
        ("Navy", "Formal"),
        ("Blue", "Casual"),
        ("Green", "Outdoor"),
    ]
    
    colors = ["Red", "Pink", "Blue", "Navy", "Green"]
    categories = ["Casual", "Formal", "Outdoor"]
    
    # Convert to numbers
    color_to_id = {color: i for i, color in enumerate(colors)}
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    
    color_ids = [color_to_id[color] for color, _ in data]
    cat_ids = [cat_to_id[cat] for _, cat in data]
    
    print("Training data:")
    for color, cat in data:
        print(f"{color} -> {cat}")
    
    print(f"\nColor IDs: {color_ids}")
    print(f"Category IDs: {cat_ids}")
    
    print("\nSTEP 2: Build model with embedding layer")
    
    # Build model
    model = keras.Sequential([
        layers.Embedding(input_dim=5,      # 5 colors
                        output_dim=3,     # 3-dimensional embeddings  
                        input_length=1),  # 1 color per input
        layers.Flatten(),
        layers.Dense(3, activation='softmax')  # 3 categories
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Model structure:")
    print("Color ID -> Embedding (3D) -> Dense -> Category Prediction")
    
    print("\nSTEP 3: Train the model")
    
    X = np.array(color_ids)
    y = np.array(cat_ids)
    
    print("Training...")
    model.fit(X, y, epochs=50, verbose=0)
    
    print("\nSTEP 4: Extract learned color embeddings")
    
    # Get embeddings
    embedding_layer = model.layers[0]
    color_embeddings = embedding_layer.get_weights()[0]
    
    print("\nLearned color embeddings:")
    print("Each color is now 3 numbers that capture its 'meaning':")
    
    for i, color in enumerate(colors):
        emb = color_embeddings[i]
        print(f"{color:5}: [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}]")
    
    print("\nSTEP 5: Check similarity")
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(color_embeddings)
    
    print("\nColor similarities (1.0 = identical, 0.0 = completely different):")
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors):
            if i < j:
                sim = similarities[i][j]
                print(f"{color1} & {color2}: {sim:.3f}")
    
    return color_embeddings

def explain_autoencoders():
    """
    Chapter 7: Autoencoders - Compression Learning
    """
    
    print("="*60)
    print("CHAPTER 7: AUTOENCODERS - COMPRESSION LEARNING")
    print("="*60)
    
    print("""
WHAT IS AN AUTOENCODER?

MOVIE SUMMARY ANALOGY:
- You watch a 2-hour movie (original data)
- You write a 1-paragraph summary (compressed version)
- Someone reads your summary and tries to retell the movie (reconstruction)
- If they can retell it well, your summary captured the important parts!

AUTOENCODER STRUCTURE:

Input -> Encoder -> Bottleneck -> Decoder -> Output
 100      50         10          50        100
features  ↓          ↓           ↓        features

WHAT EACH PART DOES:

1. ENCODER: Compresses your data
   Takes 100 product features -> reduces to 10 numbers
   Like summarizing: "casual, cotton, summer, affordable"

2. BOTTLENECK: The compressed representation (EMBEDDING!)
   Just 10 numbers that capture the product's "essence"
   This is what we want for recommendations!

3. DECODER: Tries to reconstruct original
   Takes 10 numbers -> tries to predict all 100 original features
   Like expanding summary back to full description

THE LEARNING PROCESS:

Day 1: Random compression, terrible reconstruction
Input:  [Red, Cotton, T-shirt, $25, Summer, ...]
Embedding: [0.1, 0.5, 0.8, ...]  (random)
Output: [Blue, Silk, Pants, $100, Winter, ...] <- WRONG!

Day 100: Learned compression, good reconstruction  
Input:  [Red, Cotton, T-shirt, $25, Summer, ...]
Embedding: [0.8, 0.2, 0.9, ...]  (learned)
Output: [Red, Cotton, T-shirt, $24, Summer, ...] <- GOOD!

WHY THIS WORKS:
- The embedding MUST contain important info to reconstruct well
- Similar products end up with similar embeddings
- Result: Great embeddings for recommendations!
""")

def demonstrate_autoencoder():
    """
    Chapter 8: Autoencoder Demo
    """
    
    print("="*60)
    print("CHAPTER 8: AUTOENCODER DEMONSTRATION")
    print("="*60)
    
    print("\nSTEP 1: Create sample product data")
    
    # Create sample data - 3 types of products
    np.random.seed(42)
    
    # Type 1: Summer clothes (light, bright, casual)
    summer_products = []
    for _ in range(20):
        product = np.random.normal(0, 0.1, 8)  # 8 features
        product[0:3] += 1.0  # High on summer features
        summer_products.append(product)
    
    # Type 2: Winter clothes (heavy, dark, warm)
    winter_products = []
    for _ in range(20):
        product = np.random.normal(0, 0.1, 8)
        product[3:6] += 1.0  # High on winter features  
        winter_products.append(product)
    
    # Type 3: Formal clothes (elegant, expensive, structured)
    formal_products = []
    for _ in range(20):
        product = np.random.normal(0, 0.1, 8)
        product[6:8] += 1.0  # High on formal features
        formal_products.append(product)
    
    # Combine all data
    all_products = summer_products + winter_products + formal_products
    labels = ["Summer"] * 20 + ["Winter"] * 20 + ["Formal"] * 20
    
    X = np.array(all_products)
    
    print(f"Created {len(all_products)} products with {X.shape[1]} features each")
    print("Product types: Summer, Winter, Formal")
    print(f"Data shape: {X.shape}")
    
    print("\nSTEP 2: Build autoencoder")
    
    input_dim = X.shape[1]  # 8 features
    encoding_dim = 3        # Compress to 3 dimensions
    
    # Encoder
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(5, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu', name='embedding')(encoded)
    
    # Decoder  
    decoded = layers.Dense(5, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    # Models
    autoencoder = keras.Model(input_layer, decoded)
    encoder = keras.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print(f"Autoencoder: {input_dim} -> {5} -> {encoding_dim} -> {5} -> {input_dim}")
    
    print("\nSTEP 3: Train autoencoder")
    
    print("Training to reconstruct input data...")
    history = autoencoder.fit(X, X, epochs=100, batch_size=10, verbose=0)
    
    final_loss = history.history['loss'][-1]
    print(f"Final reconstruction loss: {final_loss:.4f}")
    
    print("\nSTEP 4: Extract embeddings")
    
    # Get embeddings
    embeddings = encoder.predict(X, verbose=0)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("Each product is now represented by 3 numbers!")
    
    # Show examples
    print("\nExample embeddings:")
    for i in [5, 25, 45]:  # One from each type
        product_type = labels[i]
        emb = embeddings[i]
        print(f"{product_type:6}: [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}]")
    
    print("\nSTEP 5: Test reconstruction")
    
    # Test reconstruction
    reconstructed = autoencoder.predict(X[:3], verbose=0)
    
    print("Reconstruction quality:")
    for i in range(3):
        original = X[i]
        recon = reconstructed[i]
        error = np.mean((original - recon) ** 2)
        print(f"{labels[i]} product: Error = {error:.4f}")
    
    print("\nSTEP 6: Visualize results")
    
    # Simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot embeddings (first 2 dimensions)
    colors = {'Summer': 'red', 'Winter': 'blue', 'Formal': 'green'}
    for product_type in ['Summer', 'Winter', 'Formal']:
        mask = [label == product_type for label in labels]
        ax1.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=colors[product_type], label=product_type, alpha=0.7)
    
    ax1.set_xlabel('Embedding Dimension 1')
    ax1.set_ylabel('Embedding Dimension 2') 
    ax1.set_title('Learned Product Embeddings')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training progress
    ax2.plot(history.history['loss'])
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Reconstruction Loss')
    ax2.set_title('Training Progress')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'autoencoder_results.png'")
    
    return embeddings, autoencoder

def practical_applications():
    """
    Chapter 9: When to Use What
    """
    
    print("="*60)
    print("CHAPTER 9: PRACTICAL APPLICATIONS")
    print("="*60)
    
    print("""
NOW YOU UNDERSTAND THE METHODS - WHEN DO YOU USE WHAT?

1. EMBEDDING LAYERS:
   WHEN: You have categorical data (colors, brands, sizes)
   EXAMPLE: "Learn that Red and Pink are similar colors"
   USE FOR: Product categories, user preferences, item features

2. AUTOENCODERS:  
   WHEN: You have many features (>50) you want to compress
   EXAMPLE: "100 product features -> 10 key characteristics"
   USE FOR: Dimensionality reduction, anomaly detection, compression

3. SIMPLE NEURAL NETWORKS:
   WHEN: You have a clear prediction task
   EXAMPLE: "Predict if customer will buy this product"
   USE FOR: Classification, regression, recommendation scoring

DECISION FLOWCHART:

Do you have categorical data?
├─ YES: Use Embedding Layers
└─ NO ↓

Do you have >50 features?  
├─ YES: Use Autoencoders
└─ NO ↓

Do you need to predict something specific?
├─ YES: Use Simple Neural Network
└─ NO: Start with basic analysis

REAL BUSINESS EXAMPLES:

E-COMMERCE RECOMMENDATIONS:
1. Embedding Layers: Learn product category relationships
2. Autoencoders: Compress product features to core attributes  
3. Neural Networks: Predict purchase probability

STREAMING SERVICE:
1. Embedding Layers: Learn movie genre similarities
2. Autoencoders: Compress viewing history to user preferences
3. Neural Networks: Predict movie ratings

SOCIAL MEDIA:
1. Embedding Layers: Learn hashtag relationships
2. Autoencoders: Compress user activity to interests
3. Neural Networks: Predict content engagement
""")

def create_complete_workflow():
    """
    Chapter 10: Complete Workflow Example  
    """
    
    print("="*60)
    print("CHAPTER 10: COMPLETE WORKFLOW EXAMPLE")
    print("="*60)
    
    print("\nSCENARIO: Build recommendation system for online store")
    print("DATA: Products with categories, colors, prices")
    print("GOAL: Recommend similar products to customers")
    print()
    
    # Step 1: Create realistic data
    print("STEP 1: Create sample data")
    print("-" * 25)
    
    np.random.seed(42)
    
    products = pd.DataFrame({
        'id': range(50),
        'category': np.random.choice(['Shirt', 'Pants', 'Dress', 'Jacket'], 50),
        'color': np.random.choice(['Red', 'Blue', 'Black', 'White', 'Green'], 50),
        'price': np.random.uniform(20, 200, 50),
        'material': np.random.choice(['Cotton', 'Polyester', 'Denim'], 50)
    })
    
    print(f"Created {len(products)} products")
    print("\nSample products:")
    print(products.head())
    
    # Step 2: Prepare data for embedding
    print("\nSTEP 2: Prepare data for embedding layers")
    print("-" * 40)
    
    from sklearn.preprocessing import LabelEncoder
    
    # Encode categories
    encoders = {}
    for col in ['category', 'color', 'material']:
        encoder = LabelEncoder()
        products[f'{col}_encoded'] = encoder.fit_transform(products[col])
        encoders[col] = encoder
        print(f"{col}: {len(encoder.classes_)} unique values")
    
    # Step 3: Build embedding model
    print("\nSTEP 3: Build model with embeddings")
    print("-" * 35)
    
    # Create inputs for each categorical feature
    category_input = keras.Input(shape=(1,), name='category')
    color_input = keras.Input(shape=(1,), name='color')
    material_input = keras.Input(shape=(1,), name='material')
    price_input = keras.Input(shape=(1,), name='price')
    
    # Create embeddings
    category_emb = layers.Embedding(4, 2)(category_input)  # 4 categories -> 2D
    color_emb = layers.Embedding(5, 2)(color_input)       # 5 colors -> 2D
    material_emb = layers.Embedding(3, 2)(material_input) # 3 materials -> 2D
    
    # Flatten embeddings
    category_flat = layers.Flatten()(category_emb)
    color_flat = layers.Flatten()(color_emb)
    material_flat = layers.Flatten()(material_emb)
    
    # Combine all features
    combined = layers.Concatenate()([category_flat, color_flat, material_flat, price_input])
    
    # Create final embedding
    hidden = layers.Dense(8, activation='relu')(combined)
    product_embedding = layers.Dense(4, activation='relu', name='product_emb')(hidden)
    
    # Output for training (predict category)
    output = layers.Dense(4, activation='softmax')(product_embedding)
    
    # Create models
    full_model = keras.Model([category_input, color_input, material_input, price_input], output)
    embedding_model = keras.Model([category_input, color_input, material_input, price_input], product_embedding)
    
    full_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print("Model built: Categorical features -> Embeddings -> Product representation")
    
    # Step 4: Train model
    print("\nSTEP 4: Train the model")
    print("-" * 20)
    
    # Prepare training data
    X_train = [
        products['category_encoded'].values.reshape(-1, 1),
        products['color_encoded'].values.reshape(-1, 1), 
        products['material_encoded'].values.reshape(-1, 1),
        products['price'].values.reshape(-1, 1)
    ]
    y_train = products['category_encoded'].values
    
    print("Training...")
    full_model.fit(X_train, y_train, epochs=50, verbose=0)
    
    # Step 5: Generate embeddings
    print("\nSTEP 5: Generate product embeddings")
    print("-" * 35)
    
    product_embeddings = embedding_model.predict(X_train, verbose=0)
    
    print(f"Generated {len(product_embeddings)} product embeddings")
    print(f"Each product is now represented by {product_embeddings.shape[1]} numbers")
    
    # Step 6: Find similar products
    print("\nSTEP 6: Test recommendations")
    print("-" * 28)
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    def find_similar_products(product_id, n_similar=3):
        # Calculate similarities
        similarities = cosine_similarity([product_embeddings[product_id]], product_embeddings)[0]
        
        # Get most similar (excluding itself)
        similar_indices = similarities.argsort()[-n_similar-1:-1][::-1]
        
        return similar_indices, similarities[similar_indices]
    
    # Example recommendation
    sample_id = 0
    similar_ids, scores = find_similar_products(sample_id)
    
    print(f"Recommendations for Product {sample_id}:")
    original = products.iloc[sample_id]
    print(f"Original: {original['category']} | {original['color']} | ${original['price']:.0f}")
    
    print("\nSimilar products:")
    for idx, score in zip(similar_ids, scores):
        similar = products.iloc[idx]
        print(f"Product {idx}: {similar['category']} | {similar['color']} | ${similar['price']:.0f} (similarity: {score:.3f})")
    
    print("\nSTEP 7: Evaluate quality")
    print("-" * 22)
    
    # Check if recommendations make sense
    original_category = original['category']
    similar_categories = [products.iloc[idx]['category'] for idx in similar_ids]
    category_matches = sum(1 for cat in similar_categories if cat == original_category)
    
    print(f"Category accuracy: {category_matches}/{len(similar_ids)} = {category_matches/len(similar_ids):.1%}")
    print("SUCCESS! The model learned to recommend similar products!")
    
    return product_embeddings, products

def final_summary():
    """
    Chapter 11: Summary and Next Steps
    """
    
    print("="*60)
    print("CHAPTER 11: SUMMARY AND NEXT STEPS")
    print("="*60)
    
    print("""
CONGRATULATIONS! 🎉

You now understand the fundamentals of deep learning for embeddings!

WHAT YOU'VE LEARNED:

✓ Embeddings are "smart coordinates" that capture meaning
✓ Neural networks can learn these coordinates automatically  
✓ Embedding layers handle categorical data intelligently
✓ Autoencoders compress data while preserving information
✓ Different methods work better for different problems
✓ You can build complete recommendation systems!

KEY TAKEAWAYS:

1. START SIMPLE: Begin with basic neural networks before complex models
2. UNDERSTAND YOUR DATA: Know whether you have categories or features
3. CHOOSE THE RIGHT TOOL: Match the method to your problem
4. EVALUATE RESULTS: Always check if recommendations make sense
5. ITERATE AND IMPROVE: Start basic, then add complexity

YOUR DEEP LEARNING JOURNEY:

BEGINNER (You are here!):
✓ Understand concepts and terminology
✓ Can explain embeddings to others
✓ Know when to use different methods

INTERMEDIATE (Next 3-6 months):
• Practice with real datasets
• Learn hyperparameter tuning
• Build end-to-end projects
• Study evaluation metrics

ADVANCED (6-12 months):
• Design custom architectures
• Handle production deployment  
• Optimize for performance
• Research new techniques

RECOMMENDED NEXT STEPS:

1. PRACTICE: Use this code on your own data
2. EXPERIMENT: Try different parameters and see what happens
3. BUILD: Create a complete project from start to finish
4. LEARN: Study online courses and read research papers
5. SHARE: Show your work and get feedback from others

FINAL ADVICE:

• Every expert was once a beginner
• Don't be afraid to make mistakes - that's how you learn!
• Focus on understanding WHY things work, not just HOW
• The field evolves quickly - stay curious and keep learning
• Build a portfolio of projects to show your skills

Good luck on your deep learning journey! 🚀

REMEMBER: You now have the foundation to build amazing AI systems.
Keep practicing, stay curious, and you'll be amazed at what you can create!
""")

def run_complete_guide():
    """
    Run the complete beginner's guide
    """
    
    print("="*80)
    print("DEEP LEARNING FOR EMBEDDINGS: COMPLETE BEGINNER'S GUIDE")
    print("="*80)
    print()
    
    # Run all chapters with demonstrations
    explain_embeddings_basics()
    input("\nPress Enter to continue to the next chapter...")
    
    demonstrate_simple_example()
    input("\nPress Enter to continue...")
    
    explain_neural_networks()
    input("\nPress Enter to continue...")
    
    model, embeddings = build_simple_neural_network()
    input("\nPress Enter to continue...")
    
    explain_embedding_layers()
    input("\nPress Enter to continue...")
    
    color_embeddings = demonstrate_embedding_layer()
    input("\nPress Enter to continue...")
    
    explain_autoencoders()
    input("\nPress Enter to continue...")
    
    ae_embeddings, autoencoder = demonstrate_autoencoder()
    input("\nPress Enter to continue...")
    
    practical_applications()
    input("\nPress Enter to continue...")
    
    product_embeddings, products = create_complete_workflow()
    input("\nPress Enter for final summary...")
    
    final_summary()
    
    print("\n" + "="*80)
    print("GUIDE COMPLETE!")
    print("="*80)
    print("✓ All concepts explained with working examples")
    print("✓ Multiple hands-on demonstrations completed")
    print("✓ Ready to start your own deep learning projects!")
    print("\nThank you for learning with us! 🎓")

if __name__ == "__main__":
    # For non-interactive use, run without input prompts
    explain_embeddings_basics()
    demonstrate_simple_example()
    explain_neural_networks()
    build_simple_neural_network()
    explain_embedding_layers()
    demonstrate_embedding_layer()
    explain_autoencoders()
    demonstrate_autoencoder()
    practical_applications()
    create_complete_workflow()
    final_summary()
    
    print("\n" + "="*80)
    print("COMPLETE GUIDE FINISHED!")
    print("="*80)