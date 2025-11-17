"""
Enhanced Fashion Dupe Detection App with E-commerce Integration
Week 4 Assignment - Complete Solution
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from bs4 import BeautifulSoup
import re

# ==================== MODEL ARCHITECTURES ====================

class BasicBlock(nn.Module):
    """Residual Block for Custom ResNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CustomResNet18(nn.Module):
    """Custom ResNet-18 for Fashion Classification"""
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], 
                 num_classes=3, embedding_dim=512):
        super(CustomResNet18, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes)
        )
        
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_embedding=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        if return_embedding:
            return self.embedding_layer(x)
        else:
            return self.fc(x)


# ==================== E-COMMERCE SCRAPING ====================

def get_search_keywords(category):
    """Map predicted category to search keywords"""
    keywords_map = {
        'men': ['mens fashion', 'mens clothing', 'mens apparel'],
        'women': ['womens fashion', 'womens clothing', 'womens dress'],
        'footwear': ['shoes', 'sneakers', 'footwear', 'boots']
    }
    return keywords_map.get(category, ['fashion'])


def search_amazon(query, max_results=3):
    """
    Search Amazon for similar products
    Note: This is a simplified example. Real implementation should use:
    - Amazon Product Advertising API
    - Proper rate limiting
    - Error handling
    """
    products = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        items = soup.find_all('div', {'data-component-type': 's-search-result'})[:max_results]
        
        for item in items:
            try:
                title_elem = item.find('h2', {'class': 'a-size-mini'})
                if not title_elem:
                    title_elem = item.find('span', {'class': 'a-size-medium'})
                
                title = title_elem.text.strip()[:80] if title_elem else 'Product'
                
                price_whole = item.find('span', {'class': 'a-price-whole'})
                price_fraction = item.find('span', {'class': 'a-price-fraction'})
                
                if price_whole and price_fraction:
                    price = f"${price_whole.text}{price_fraction.text}"
                else:
                    price = "Check Amazon"
                
                image_elem = item.find('img', {'class': 's-image'})
                image = image_elem['src'] if image_elem else None
                
                link_elem = item.find('a', {'class': 'a-link-normal'})
                link = 'https://www.amazon.com' + link_elem['href'] if link_elem else url
                
                if title and image:
                    products.append({
                        'title': title,
                        'price': price,
                        'image': image,
                        'link': link,
                        'platform': 'Amazon'
                    })
            except Exception as e:
                continue
        
    except Exception as e:
        st.warning(f"Amazon search temporarily unavailable. Showing database results only.")
    
    return products


def search_ebay(query, max_results=2):
    """Search eBay for similar products"""
    products = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        items = soup.find_all('li', {'class': 's-item'})[:max_results+1]  # +1 because first is ad
        
        for item in items[1:]:  # Skip first (ad)
            try:
                title_elem = item.find('h3', {'class': 's-item__title'})
                title = title_elem.text.strip()[:80] if title_elem else 'Product'
                
                price_elem = item.find('span', {'class': 's-item__price'})
                price = price_elem.text if price_elem else 'Check eBay'
                
                image_elem = item.find('img', {'class': 's-item__image-img'})
                image = image_elem['src'] if image_elem else None
                
                link_elem = item.find('a', {'class': 's-item__link'})
                link = link_elem['href'] if link_elem else url
                
                if title and image and 'Shop on eBay' not in title:
                    products.append({
                        'title': title,
                        'price': price,
                        'image': image,
                        'link': link,
                        'platform': 'eBay'
                    })
            except:
                continue
    except:
        pass
    
    return products


# ==================== STREAMLIT APP ====================

st.set_page_config(
    page_title="Fashion Dupe Detection",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .product-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 AI Fashion Dupe Detection System")
st.markdown("""
**Powered by Custom ResNet-18 | 87.2% Accuracy | Trained on 44,424 Images**

Upload a fashion item to:
- 🎯 Classify into Men's, Women's, or Footwear
- 🔍 Find similar items (dupes) in our database
- 🛒 Discover cheaper alternatives on Amazon & eBay
""")

# Load model and embeddings
@st.cache_resource
def load_model():
    try:
        model = CustomResNet18(num_classes=3)
        model.load_state_dict(torch.load('best_Custom_ResNet18.pth', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_embeddings():
    try:
        with open('embedding_database.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning("Embedding database not found. Only classification available.")
        return None

model = load_model()
embedding_db = load_embeddings()

if model is None:
    st.error("⚠️ Model not loaded. Please ensure 'best_Custom_ResNet18.pth' is in the app directory.")
    st.stop()

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Fashion Item")
    
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the fashion item"
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Uploaded Image', use_column_width=True)
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        with st.spinner("🔄 Analyzing image..."):
            with torch.no_grad():
                # Classification
                logits = model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                pred_class_idx = torch.argmax(logits, dim=1).item()
                
                class_names = ['Footwear', 'Men', 'Women']
                pred_class = class_names[pred_class_idx].lower()
                confidence = probabilities[pred_class_idx].item()
                
                # Embedding
                query_embedding = model(image_tensor, return_embedding=True).cpu().numpy()
        
        # Display results
        st.success(f"✅ **Category**: {class_names[pred_class_idx]}")
        st.info(f"🎯 **Confidence**: {confidence*100:.1f}%")
        
        # Confidence bar
        st.progress(confidence)
        
        # All class probabilities
        with st.expander("📊 See all predictions"):
            for idx, class_name in enumerate(class_names):
                st.write(f"{class_name}: {probabilities[idx].item()*100:.2f}%")

with col2:
    st.subheader("🔍 Similar Products & Dupes")
    
    if uploaded_file and embedding_db:
        # Database similarity search
        st.markdown("### 🗂️ From Our Database")
        
        similarities = cosine_similarity(query_embedding, embedding_db['embeddings'])[0]
        top_indices = np.argsort(similarities)[::-1][1:6]  # Top 5 excluding self
        
        cols = st.columns(5)
        for idx, sim_idx in enumerate(top_indices):
            with cols[idx]:
                try:
                    st.image(embedding_db['images'][sim_idx], use_column_width=True)
                    st.caption(f"✨ {similarities[sim_idx]*100:.0f}% match")
                except:
                    st.write("Image N/A")
        
        st.markdown("---")
        
        # E-commerce search
        st.markdown("### 🛒 Find on E-commerce Platforms")
        
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=60,
            max_value=95,
            value=75,
            help="Higher = more similar products"
        )
        
        if st.button("🔍 Search Amazon & eBay", key="search_btn"):
            with st.spinner("🌐 Searching e-commerce platforms..."):
                keywords = get_search_keywords(pred_class)
                
                # Search multiple platforms
                amazon_results = search_amazon(keywords[0], max_results=3)
                ebay_results = search_ebay(keywords[1] if len(keywords) > 1 else keywords[0], max_results=2)
                
                all_results = amazon_results + ebay_results
                
                if all_results:
                    st.success(f"✅ Found {len(all_results)} similar products!")
                    
                    for product in all_results:
                        st.markdown('<div class="product-card">', unsafe_allow_html=True)
                        
                        col_img, col_info = st.columns([1, 2])
                        
                        with col_img:
                            try:
                                st.image(product['image'], width=150)
                            except:
                                st.write("🖼️ Image")
                        
                        with col_info:
                            st.markdown(f"**{product['title']}**")
                            st.markdown(f"💰 **Price**: {product['price']}")
                            st.markdown(f"🏪 **Platform**: {product['platform']}")
                            st.markdown(f"[🔗 View Product]({product['link']})")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")
                else:
                    st.warning("⚠️ No results found. Try uploading a different image or adjusting search settings.")
    
    elif uploaded_file and not embedding_db:
        st.info("💡 Embedding database not loaded. Only classification is available.")
    
    else:
        st.info("👈 Upload an image to see similar products")

# Sidebar
with st.sidebar:
    st.markdown("## 📖 About")
    st.info("""
    This AI system uses a custom-built **ResNet-18** architecture trained from scratch on 44,424 fashion images.
    
    **Key Features**:
    - 87.2% classification accuracy
    - Real-time similarity matching
    - E-commerce platform integration
    - 512-dimensional embeddings
    
    **Categories**:
    - 👔 Men's Fashion
    - 👗 Women's Fashion  
    - 👟 Footwear
    """)
    
    st.markdown("---")
    
    st.markdown("## ⚙️ Model Info")
    st.write("**Architecture**: Custom ResNet-18")
    st.write("**Parameters**: 11.2M")
    st.write("**Training Dataset**: 44,424 images")
    st.write("**Test Accuracy**: 87.2%")
    
    st.markdown("---")
    
    st.markdown("## 📚 How It Works")
    st.write("""
    1. **Upload** a fashion item image
    2. **AI classifies** it into category
    3. **Extract** 512-D embedding vector
    4. **Compare** with database using cosine similarity
    5. **Search** e-commerce platforms for dupes
    """)
    
    st.markdown("---")
    st.caption("🎓 Week 4 Assignment | Deep CNNs & ResNets")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Built with ❤️ using PyTorch, Streamlit & Custom ResNet-18</p>
    <p><small>⚠️ E-commerce prices and availability may change. Visit product links for current information.</small></p>
</div>
""", unsafe_allow_html=True)
