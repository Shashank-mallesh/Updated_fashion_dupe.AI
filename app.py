import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(
    page_title="Fashion Dupe Detection",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Fashion Dupe Detection System")
st.markdown("Upload fashion item images to classify them and detect duplicates")

# Simple mock classification function
def classify_image(image):
    """Mock classification based on image properties"""
    img_array = np.array(image)
    
    # Simple logic based on image characteristics
    if len(img_array.shape) == 3:
        height, width, _ = img_array.shape
        aspect_ratio = width / height
        
        if aspect_ratio < 0.7:
            return "footwear", 0.85
        elif aspect_ratio > 1.3:
            return "men", 0.78
        else:
            return "women", 0.82
    return "unknown", 0.5

# Simple mock similarity function
def calculate_similarity(img1, img2):
    """Mock similarity calculation"""
    return np.random.uniform(0.6, 0.9)

# Main app
tab1, tab2 = st.tabs(["📸 Image Classification", "🔍 Dupe Detection"])

with tab1:
    st.header("Classify Fashion Items")
    
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("Analyzing..."):
                category, confidence = classify_image(image)
            
            st.success(f"**Predicted Category:** {category.upper()}")
            st.info(f"**Confidence:** {confidence:.1%}")
            
            # Show confidence bars
            categories = ['footwear', 'men', 'women']
            confidences = [0.1, 0.2, 0.7] if category == 'women' else \
                         [0.7, 0.2, 0.1] if category == 'footwear' else \
                         [0.2, 0.7, 0.1]
            
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.barh(categories, [c * 100 for c in confidences], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_xlim(0, 100)
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Classification Probabilities')
            
            for bar, conf in zip(bars, confidences):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{conf*100:.1f}%', 
                       va='center', fontweight='bold')
            
            st.pyplot(fig)

with tab2:
    st.header("Detect Duplicate Items")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img1 = st.file_uploader("Upload original item", type=['png', 'jpg', 'jpeg'], key="img1")
        if img1:
            st.image(Image.open(img1), caption="Original Item", use_container_width=True)
    
    with col2:
        img2 = st.file_uploader("Upload potential dupe", type=['png', 'jpg', 'jpeg'], key="img2")
        if img2:
            st.image(Image.open(img2), caption="Potential Dupe", use_container_width=True)
    
    threshold = st.slider("Similarity Threshold", 50, 95, 75, help="Higher values mean stricter matching")
    
    if img1 and img2:
        if st.button("🔍 Compare Images", type="primary"):
            with st.spinner("Comparing images..."):
                similarity = calculate_similarity(img1, img2)
                is_dupe = similarity >= (threshold / 100)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cat1, conf1 = classify_image(Image.open(img1))
                    st.metric("Item 1 Category", cat1.upper(), f"{conf1:.1%}")
                
                with col2:
                    st.metric("Similarity", f"{similarity:.1%}", 
                             "DUPE! ✅" if is_dupe else "Not Dupe ❌")
                
                with col3:
                    cat2, conf2 = classify_image(Image.open(img2))
                    st.metric("Item 2 Category", cat2.upper(), f"{conf2:.1%}")
                
                if is_dupe:
                    st.success(f"🎯 **DUPE DETECTED!** {similarity:.1%} similarity (above {threshold}% threshold)")
                    st.balloons()
                else:
                    st.warning(f"⚠️ **Not a dupe.** {similarity:.1%} similarity (below {threshold}% threshold)")

st.markdown("---")
st.caption("Fashion Dupe Detection Demo | Built with Streamlit")
