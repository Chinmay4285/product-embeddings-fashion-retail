import pandas as pd
import numpy as np
import random
from itertools import product
from datetime import datetime, timedelta

def generate_fashion_dataset(n_products=2000, seed=42):
    """
    Generate a realistic fashion retail dataset similar to GAP's catalog structure.
    
    Parameters:
    - n_products: Number of unique SKUs to generate
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with product attributes
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Define realistic fashion categories and attributes
    categories = {
        'Tops': ['T-Shirts', 'Blouses', 'Sweaters', 'Hoodies', 'Tank Tops', 'Polo Shirts'],
        'Bottoms': ['Jeans', 'Chinos', 'Shorts', 'Leggings', 'Skirts', 'Dress Pants'],
        'Dresses': ['Casual Dresses', 'Formal Dresses', 'Maxi Dresses', 'Mini Dresses'],
        'Outerwear': ['Jackets', 'Coats', 'Blazers', 'Cardigans', 'Vests'],
        'Activewear': ['Athletic Tops', 'Athletic Bottoms', 'Sports Bras', 'Activewear Sets'],
        'Accessories': ['Belts', 'Scarves', 'Hats', 'Bags', 'Jewelry']
    }
    
    divisions = ['Women', 'Men', 'Kids', 'Baby']
    genders = ['Female', 'Male', 'Unisex']
    sizes = {
        'Women': ['XS', 'S', 'M', 'L', 'XL', 'XXL', '0', '2', '4', '6', '8', '10', '12', '14', '16'],
        'Men': ['XS', 'S', 'M', 'L', 'XL', 'XXL', '28', '30', '32', '34', '36', '38', '40', '42'],
        'Kids': ['2T', '3T', '4T', '5T', 'XS', 'S', 'M', 'L', 'XL'],
        'Baby': ['0-3M', '3-6M', '6-9M', '9-12M', '12-18M', '18-24M']
    }
    
    colors = [
        'Black', 'White', 'Navy', 'Gray', 'Beige', 'Brown', 'Khaki',
        'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink',
        'Burgundy', 'Olive', 'Charcoal', 'Cream', 'Denim', 'Coral',
        'Mint', 'Lavender', 'Sage', 'Dusty Rose', 'Camel'
    ]
    
    # Base price ranges by category (min, max)
    price_ranges = {
        'Tops': (15, 80),
        'Bottoms': (25, 120),
        'Dresses': (35, 150),
        'Outerwear': (50, 300),
        'Activewear': (20, 90),
        'Accessories': (10, 60)
    }
    
    products = []
    sku_counter = 1000000  # Starting SKU number
    
    for _ in range(n_products):
        # Select random category and subcategory
        category = random.choice(list(categories.keys()))
        subcategory = random.choice(categories[category])
        
        # Select division and corresponding gender
        division = random.choice(divisions)
        if division == 'Women':
            gender = 'Female'
        elif division == 'Men':
            gender = 'Male'
        else:  # Kids/Baby
            gender = random.choice(['Male', 'Female', 'Unisex'])
        
        # Select size based on division
        size = random.choice(sizes[division])
        
        # Select color
        color = random.choice(colors)
        
        # Generate price with some variation
        base_min, base_max = price_ranges[category]
        
        # Add premium for certain subcategories
        premium_multiplier = 1.0
        if subcategory in ['Formal Dresses', 'Blazers', 'Coats']:
            premium_multiplier = 1.3
        elif subcategory in ['Designer', 'Premium']:
            premium_multiplier = 1.5
            
        price = round(random.uniform(base_min, base_max) * premium_multiplier, 2)
        
        # Generate inventory with realistic distribution
        # Most items have moderate inventory, some are low stock, few are overstocked
        inventory_distribution = random.random()
        if inventory_distribution < 0.1:  # 10% low stock
            inventory = random.randint(0, 5)
        elif inventory_distribution < 0.8:  # 70% normal stock
            inventory = random.randint(10, 100)
        else:  # 20% high stock
            inventory = random.randint(100, 500)
        
        # Generate SKU
        sku = f"GAP{sku_counter}"
        sku_counter += 1
        
        # Create product name
        product_name = f"{color} {subcategory}"
        if random.random() < 0.3:  # 30% chance of additional descriptor
            descriptors = ['Classic', 'Vintage', 'Modern', 'Essential', 'Premium', 'Casual', 'Formal']
            product_name = f"{random.choice(descriptors)} {product_name}"
        
        # Generate additional features for embedding richness
        seasonality = random.choice(['Spring/Summer', 'Fall/Winter', 'Year-Round'])
        material = random.choice(['Cotton', 'Polyester', 'Denim', 'Wool', 'Linen', 'Silk', 'Blend'])
        fit = random.choice(['Regular', 'Slim', 'Relaxed', 'Oversized', 'Fitted'])
        
        # Create launch date (within last 2 years)
        launch_date = datetime.now() - timedelta(days=random.randint(0, 730))
        
        product = {
            'SKU': sku,
            'Product_Name': product_name,
            'Category': category,
            'Subcategory': subcategory,
            'Division': division,
            'Gender': gender,
            'Size': size,
            'Color': color,
            'Price': price,
            'Inventory': inventory,
            'Seasonality': seasonality,
            'Material': material,
            'Fit': fit,
            'Launch_Date': launch_date.strftime('%Y-%m-%d'),
            'Days_Since_Launch': (datetime.now() - launch_date).days
        }
        
        products.append(product)
    
    df = pd.DataFrame(products)
    
    # Add some derived features that might be useful for embeddings
    df['Price_Tier'] = pd.cut(df['Price'], bins=[0, 25, 50, 100, 200, float('inf')], 
                             labels=['Budget', 'Value', 'Mid-Range', 'Premium', 'Luxury'])
    
    df['Inventory_Status'] = pd.cut(df['Inventory'], bins=[-1, 5, 20, 100, float('inf')],
                                   labels=['Low Stock', 'Limited', 'In Stock', 'Overstocked'])
    
    df['Product_Age_Category'] = pd.cut(df['Days_Since_Launch'], 
                                       bins=[-1, 30, 90, 365, float('inf')],
                                       labels=['New', 'Recent', 'Established', 'Legacy'])
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating fashion retail dataset...")
    fashion_df = generate_fashion_dataset(n_products=2000, seed=42)
    
    # Save to CSV
    fashion_df.to_csv('fashion_retail_dataset.csv', index=False)
    print(f"Dataset saved with {len(fashion_df)} products")
    
    # Display basic info
    print("\nDataset Overview:")
    print(f"Shape: {fashion_df.shape}")
    print(f"Columns: {list(fashion_df.columns)}")
    
    print("\nSample products:")
    print(fashion_df.head())
    
    print("\nCategory distribution:")
    print(fashion_df['Category'].value_counts())
    
    print("\nPrice statistics:")
    print(fashion_df['Price'].describe())