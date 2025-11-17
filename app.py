# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import time
import json

# Set page config
st.set_page_config(
    page_title="Fashion Dupe Finder",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .price-tag {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e74c3c;
    }
    .rating {
        color: #f39c12;
    }
    .match-percentage {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
    }
    .high-match {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-match {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-match {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class FashionDupeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FashionDupeClassifier, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the trained model"""
    model = FashionDupeClassifier(num_classes=2)
    try:
        model.load_state_dict(torch.load('fashion-data/models/best_fashion_dupe_model_30_epochs.pth', 
                                       map_location=torch.device('cpu')))
        model.eval()
        return model
    except:
        st.error("Model file not found. Please ensure the model is trained and saved in the correct location.")
        return None

def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_similarity_score(model, image):
    """Get similarity score from model"""
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, 1)
        similarity_score = probabilities[0][1].item()  # Probability it's a dupe
    return similarity_score

# Mock e-commerce data - In a real application, you'd query actual APIs
def get_mock_product_data():
    """Generate mock product data for demonstration"""
    products = []
    
    # Nike Air Force 1 example
    base_product = {
        'name': 'Nike Air Force 1 White',
        'original_price': 120.00,
        'brand': 'Nike',
        'category': 'Sneakers',
        'description': 'Classic white sneakers with durable construction and comfortable cushioning.'
    }
    
    # Similar products from different retailers
    similar_products = [
        {
            'name': 'Air Force 1 Style White Sneakers',
            'retailer': 'Amazon Fashion',
            'price': 45.99,
            'rating': 4.2,
            'reviews': 1247,
            'match_percentage': 92,
            'url': 'https://amazon.com/fashion-sneakers',
            'image_url': 'https://via.placeholder.com/200x200/FF6B6B/FFFFFF?text=Amazon+Version'
        },
        {
            'name': 'Premium White Leather Sneakers',
            'retailer': 'Walmart',
            'price': 39.99,
            'rating': 4.0,
            'reviews': 892,
            'match_percentage': 88,
            'url': 'https://walmart.com/premium-sneakers',
            'image_url': 'https://via.placeholder.com/200x200/4ECDC4/FFFFFF?text=Walmart+Version'
        },
        {
            'name': 'Classic White Athletic Shoes',
            'retailer': 'Target',
            'price': 49.99,
            'rating': 4.3,
            'reviews': 567,
            'match_percentage': 85,
            'url': 'https://target.com/athletic-shoes',
            'image_url': 'https://via.placeholder.com/200x200/45B7D1/FFFFFF?text=Target+Version'
        },
        {
            'name': 'Urban White Casual Sneakers',
            'retailer': 'AliExpress',
            'price': 28.50,
            'rating': 3.8,
            'reviews': 2341,
            'match_percentage': 78,
            'url': 'https://aliexpress.com/urban-sneakers',
            'image_url': 'https://via.placeholder.com/200x200/F7DC6F/FFFFFF?text=AliExpress+Version'
        },
        {
            'name': 'Designer Inspired White Shoes',
            'retailer': 'Shein',
            'price': 32.99,
            'rating': 4.1,
            'reviews': 1789,
            'match_percentage': 82,
            'url': 'https://shein.com/designer-shoes',
            'image_url': 'https://via.placeholder.com/200x200/BB8FCE/FFFFFF?text=Shein+Version'
        }
    ]
    
    return base_product, similar_products

def get_match_color_class(percentage):
    """Get CSS class based on match percentage"""
    if percentage >= 90:
        return "high-match"
    elif percentage >= 75:
        return "medium-match"
    else:
        return "low-match"

def main():
    # Header
    st.markdown('<h1 class="main-header">👟 Fashion Dupe Finder</h1>', unsafe_allow_html=True)
    st.markdown("### Find affordable alternatives to your favorite fashion items")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "Upload an image of any fashion product to find similar, more affordable alternatives "
        "from various e-commerce platforms with price comparisons and customer reviews."
    )
    
    st.sidebar.title("How it works")
    st.sidebar.markdown("""
    1. Upload an image of your desired product
    2. Our AI analyzes the product features
    3. Find similar products across multiple retailers
    4. Compare prices, ratings, and reviews
    5. Save money on your fashion purchases!
    """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Product Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Product", use_column_width=True)
            
            # Load model
            with st.spinner("Analyzing product..."):
                model = load_model()
                if model:
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Get similarity score (mock for now - in real app, use actual model inference)
                    similarity_score = 0.89  # Mock score
                    
                    st.success(f"Product analysis complete!")
                    
                    # Display product info
                    base_product, similar_products = get_mock_product_data()
                    
                    st.subheader("Original Product Info")
                    st.write(f"**Name:** {base_product['name']}")
                    st.write(f"**Brand:** {base_product['brand']}")
                    st.write(f"**Category:** {base_product['category']}")
                    st.write(f"**Original Price:** ${base_product['original_price']:.2f}")
                    st.write(f"**Description:** {base_product['description']}")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Similar Products Found")
            st.write("Here are affordable alternatives to your product:")
            
            # Display similar products
            base_product, similar_products = get_mock_product_data()
            
            for product in similar_products:
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    
                    with col_a:
                        st.image(product['image_url'], width=100)
                    
                    with col_b:
                        st.write(f"**{product['name']}**")
                        st.write(f"**Retailer:** {product['retailer']}")
                        
                        # Price comparison
                        original_price = base_product['original_price']
                        savings = original_price - product['price']
                        savings_percentage = (savings / original_price) * 100
                        
                        st.write(f"**Price:** ${product['price']:.2f}")
                        st.write(f"💰 **You save: ${savings:.2f} ({savings_percentage:.1f}%)**")
                        
                        # Rating
                        st.write(f"⭐ **Rating:** {product['rating']}/5 ({product['reviews']} reviews)")
                    
                    with col_c:
                        match_class = get_match_color_class(product['match_percentage'])
                        st.markdown(f'<div class="match-percentage {match_class}">{product["match_percentage"]}% Match</div>', 
                                  unsafe_allow_html=True)
                        
                        if st.button("View Product", key=product['name']):
                            st.write(f"Redirecting to {product['retailer']}...")
                            # In a real app, this would redirect to the actual product page
                    
                    st.markdown("---")
            
            # Summary statistics
            st.subheader("💰 Price Comparison Summary")
            prices = [p['price'] for p in similar_products]
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                st.metric("Original Price", f"${base_product['original_price']:.2f}")
            with col_y:
                st.metric("Average Alternative", f"${avg_price:.2f}")
            with col_z:
                st.metric("Lowest Price Found", f"${min_price:.2f}")
            
            # Savings visualization
            st.subheader("📊 Potential Savings")
            savings_data = {
                'Retailer': [p['retailer'] for p in similar_products],
                'Price': [p['price'] for p in similar_products],
                'Savings': [base_product['original_price'] - p['price'] for p in similar_products],
                'Match %': [p['match_percentage'] for p in similar_products]
            }
            
            df = pd.DataFrame(savings_data)
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("👆 Upload a product image to find affordable alternatives!")
            st.image("https://via.placeholder.com/600x400/3498DB/FFFFFF?text=Upload+Product+Image+to+Start", 
                    use_column_width=True)

if __name__ == "__main__":
    main()
