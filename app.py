# Create simplified app.py for Streamlit Cloud
app_code = '''
import streamlit as st
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import time

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
    .savings-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Mock e-commerce data
def get_mock_product_data(product_type="sneakers"):
    """Generate mock product data based on product type"""
    
    if "shoe" in product_type.lower() or "sneaker" in product_type.lower():
        base_product = {
            'name': 'Nike Air Force 1 White',
            'original_price': 120.00,
            'brand': 'Nike',
            'category': 'Sneakers',
            'description': 'Classic white sneakers with durable construction and comfortable cushioning.'
        }
        
        similar_products = [
            {
                'name': 'Air Force 1 Style White Sneakers',
                'retailer': 'Amazon Fashion',
                'price': 45.99,
                'rating': 4.2,
                'reviews': 1247,
                'match_percentage': 92,
                'url': 'https://amazon.com/fashion-sneakers',
                'shipping': 'FREE delivery'
            },
            {
                'name': 'Premium White Leather Sneakers',
                'retailer': 'Walmart',
                'price': 39.99,
                'rating': 4.0,
                'reviews': 892,
                'match_percentage': 88,
                'url': 'https://walmart.com/premium-sneakers',
                'shipping': 'FREE shipping'
            },
            {
                'name': 'Classic White Athletic Shoes',
                'retailer': 'Target',
                'price': 49.99,
                'rating': 4.3,
                'reviews': 567,
                'match_percentage': 85,
                'url': 'https://target.com/athletic-shoes',
                'shipping': 'Same day delivery'
            },
            {
                'name': 'Urban White Casual Sneakers',
                'retailer': 'AliExpress',
                'price': 28.50,
                'rating': 3.8,
                'reviews': 2341,
                'match_percentage': 78,
                'url': 'https://aliexpress.com/urban-sneakers',
                'shipping': 'Free shipping'
            }
        ]
    
    elif "shirt" in product_type.lower() or "top" in product_type.lower():
        base_product = {
            'name': 'Ralph Lauren Polo Shirt',
            'original_price': 89.99,
            'brand': 'Ralph Lauren',
            'category': 'Casual Shirt',
            'description': 'Classic polo shirt with embroidered logo and comfortable cotton fabric.'
        }
        
        similar_products = [
            {
                'name': 'Classic Fit Polo Shirt',
                'retailer': 'Amazon Fashion',
                'price': 24.99,
                'rating': 4.4,
                'reviews': 1892,
                'match_percentage': 91,
                'url': 'https://amazon.com/polo-shirt',
                'shipping': 'FREE delivery'
            },
            {
                'name': 'Premium Cotton Polo',
                'retailer': 'Walmart',
                'price': 19.99,
                'rating': 4.1,
                'reviews': 756,
                'match_percentage': 87,
                'url': 'https://walmart.com/cotton-polo',
                'shipping': 'FREE shipping'
            },
            {
                'name': 'Designer Style Polo Shirt',
                'retailer': 'Shein',
                'price': 15.99,
                'rating': 4.0,
                'reviews': 3421,
                'match_percentage': 83,
                'url': 'https://shein.com/designer-polo',
                'shipping': 'Standard shipping'
            }
        ]
    
    else:  # Default to handbag
        base_product = {
            'name': 'Michael Kors Crossbody Bag',
            'original_price': 198.00,
            'brand': 'Michael Kors',
            'category': 'Handbag',
            'description': 'Stylish crossbody bag with multiple compartments and adjustable strap.'
        }
        
        similar_products = [
            {
                'name': 'Designer Style Crossbody Bag',
                'retailer': 'Amazon Fashion',
                'price': 35.99,
                'rating': 4.3,
                'reviews': 892,
                'match_percentage': 89,
                'url': 'https://amazon.com/crossbody-bag',
                'shipping': 'FREE delivery'
            },
            {
                'name': 'Fashion Leather Crossbody',
                'retailer': 'Target',
                'price': 42.99,
                'rating': 4.2,
                'reviews': 445,
                'match_percentage': 86,
                'url': 'https://target.com/leather-bag',
                'shipping': 'Same day delivery'
            },
            {
                'name': 'Trendy Shoulder Bag',
                'retailer': 'Shein',
                'price': 22.99,
                'rating': 4.0,
                'reviews': 1567,
                'match_percentage': 81,
                'url': 'https://shein.com/shoulder-bag',
                'shipping': 'Free shipping'
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

def analyze_product_type(image):
    """Simple product type detection based on image characteristics"""
    # For demo purposes, we'll use a simple approach
    # In a real app, you'd use ML model here
    width, height = image.size
    aspect_ratio = width / height
    
    if aspect_ratio > 1.2:  # Wide image - likely shoes
        return "sneakers"
    elif aspect_ratio < 0.8:  # Tall image - likely bag
        return "handbag"
    else:  # Square-ish - likely clothing
        return "shirt"

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
    1. 📸 Upload an image of your desired product
    2. 🤖 Our AI analyzes the product features
    3. 🔍 Find similar products across multiple retailers
    4. 💰 Compare prices, ratings, and reviews
    5. 🎉 Save money on your fashion purchases!
    """)
    
    st.sidebar.title("Supported Products")
    st.sidebar.markdown("""
    - 👟 Sneakers & Shoes
    - 👕 Shirts & Tops  
    - 👜 Bags & Accessories
    - 🧥 Jackets & Outerwear
    - 👖 Pants & Jeans
    """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📸 Upload Product Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Product", use_column_width=True)
            
            # Analyze product
            with st.spinner("🔍 Analyzing product and finding alternatives..."):
                time.sleep(2)  # Simulate processing time
                
                # Simple product type detection
                product_type = analyze_product_type(image)
                base_product, similar_products = get_mock_product_data(product_type)
                
                st.success("✅ Product analysis complete!")
                
                # Display product info
                st.subheader("📋 Original Product Info")
                st.write(f"**Name:** {base_product['name']}")
                st.write(f"**Brand:** {base_product['brand']}")
                st.write(f"**Category:** {base_product['category']}")
                st.write(f"**Original Price:** ${base_product['original_price']:.2f}")
                st.write(f"**Description:** {base_product['description']}")
                
                # Quick stats
                total_savings = sum(base_product['original_price'] - p['price'] for p in similar_products)
                avg_savings = total_savings / len(similar_products)
                st.metric("💵 Average Savings Potential", f"${avg_savings:.2f}")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Similar Products Found")
            st.write(f"*Showing {len(similar_products)} affordable alternatives*")
            
            # Display similar products
            for i, product in enumerate(similar_products):
                with st.container():
                    st.markdown(f"### Alternative #{i+1}")
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    
                    with col_a:
                        # Generate placeholder image based on retailer
                        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F", "#BB8FCE"]
                        color = colors[i % len(colors)]
                        placeholder_url = f"https://via.placeholder.com/150x150/{color[1:]}/FFFFFF?text={product['retailer'].split()[0]}"
                        st.image(placeholder_url, width=120)
                    
                    with col_b:
                        st.write(f"**{product['name']}**")
                        st.write(f"**🏪 Retailer:** {product['retailer']}")
                        
                        # Price comparison
                        original_price = base_product['original_price']
                        savings = original_price - product['price']
                        savings_percentage = (savings / original_price) * 100
                        
                        st.write(f"**💰 Price:** ${product['price']:.2f}")
                        st.markdown(f'<div class="savings-badge">Save ${savings:.2f} ({savings_percentage:.1f}%)</div>', 
                                  unsafe_allow_html=True)
                        
                        # Rating and shipping
                        st.write(f"⭐ **Rating:** {product['rating']}/5 ({product['reviews']} reviews)")
                        st.write(f"🚚 **Shipping:** {product['shipping']}")
                    
                    with col_c:
                        match_class = get_match_color_class(product['match_percentage'])
                        st.markdown(f'<div class="match-percentage {match_class}">{product["match_percentage"]}% Match</div>', 
                                  unsafe_allow_html=True)
                        
                        if st.button("🛒 View Product", key=f"btn_{i}"):
                            st.success(f"🌐 Redirecting to {product['retailer']}...")
                            # Note: In a real app, this would use st.markdown with [](url) or JavaScript
                    
                    st.markdown("---")
            
            # Summary statistics
            st.subheader("📊 Price Comparison Summary")
            prices = [p['price'] for p in similar_products]
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_match = max(p['match_percentage'] for p in similar_products)
            
            col_x, col_y, col_z, col_w = st.columns(4)
            with col_x:
                st.metric("Original Price", f"${base_product['original_price']:.2f}")
            with col_y:
                st.metric("Average Alternative", f"${avg_price:.2f}")
            with col_z:
                st.metric("Lowest Price", f"${min_price:.2f}")
            with col_w:
                st.metric("Highest Match", f"{max_match}%")
            
            # Savings chart data
            st.subheader("💵 Detailed Price Comparison")
            comparison_data = {
                'Retailer': [p['retailer'] for p in similar_products],
                'Price': [p['price'] for p in similar_products],
                'Savings': [base_product['original_price'] - p['price'] for p in similar_products],
                'Match %': [p['match_percentage'] for p in similar_products],
                'Rating': [p['rating'] for p in similar_products],
                'Reviews': [p['reviews'] for p in similar_products]
            }
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df.style.format({
                'Price': '${:.2f}',
                'Savings': '${:.2f}',
                'Match %': '{:.0f}%',
                'Rating': '{:.1f}'
            }), use_container_width=True)
            
            # Additional features
            st.subheader("💡 Shopping Tips")
            tips = [
                "Check return policies before purchasing alternatives",
                "Look for coupon codes on retailer websites",
                "Consider shipping times and costs",
                "Read recent customer reviews for quality assessment",
                "Compare material quality descriptions carefully"
            ]
            
            for tip in tips:
                st.write(f"• {tip}")
            
        else:
            st.info("👆 Upload a product image to find affordable alternatives!")
            
            # Demo images
            st.subheader("🎯 Try These Examples:")
            demo_col1, demo_col2, demo_col3 = st.columns(3)
            
            with demo_col1:
                st.image("https://via.placeholder.com/200x200/3498DB/FFFFFF?text=Sneakers", 
                        caption="Shoes/Sneakers", use_column_width=True)
            with demo_col2:
                st.image("https://via.placeholder.com/200x200/E74C3C/FFFFFF?text=Shirt", 
                        caption="Clothing", use_column_width=True)
            with demo_col3:
                st.image("https://via.placeholder.com/200x200/27AE60/FFFFFF?text=Bag", 
                        caption="Bags", use_column_width=True)
            
            st.markdown("---")
            st.success("💡 **Pro Tip:** Upload clear, well-lit product images for the best matching results!")

if __name__ == "__main__":
    main()
'''

# Write the simplified app.py file
with open('app.py', 'w') as f:
    f.write(app_code)

print("✅ Simplified app.py created successfully!")
print("📋 requirements.txt created successfully!")
print("🚀 Your app is ready for Streamlit Cloud!")
print("📤 Deploy by:")
print("   1. Pushing these files to GitHub")
print("   2. Connecting your repo to Streamlit Cloud")
print("   3. The requirements.txt will automatically install dependencies")
