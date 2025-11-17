import streamlit as st
import numpy as np
from PIL import Image

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
            
            # Show confidence bars using Streamlit native components
            st.subheader("Classification Confidence")
            
            categories = ['footwear', 'men', 'women']
            if category == 'women':
                confidences = [10, 20, 70]
            elif category == 'footwear':
                confidences = [70, 20, 10]
            else:  # men
                confidences = [20, 70, 10]
            
            for cat, conf in zip(categories, confidences):
                st.write(f"**{cat.upper()}**: {conf}%")
                st.progress(conf / 100)

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
                similarity_percent = similarity * 100
                is_dupe = similarity_percent >= threshold
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cat1, conf1 = classify_image(Image.open(img1))
                    st.metric("Item 1 Category", cat1.upper(), f"{conf1:.1%}")
                
                with col2:
                    st.metric("Similarity", f"{similarity_percent:.1f}%", 
                             "DUPE! ✅" if is_dupe else "Not Dupe ❌")
                
                with col3:
                    cat2, conf2 = classify_image(Image.open(img2))
                    st.metric("Item 2 Category", cat2.upper(), f"{conf2:.1%}")
                
                # Visual similarity gauge using native Streamlit
                st.subheader("Similarity Gauge")
                st.write(f"Current similarity: **{similarity_percent:.1f}%**")
                st.write(f"Threshold: **{threshold}%**")
                
                # Create a simple gauge using progress bar
                if is_dupe:
                    st.progress(similarity_percent / 100)
                    st.success(f"🎯 **DUPE DETECTED!** {similarity_percent:.1f}% similarity (above {threshold}% threshold)")
                    st.balloons()
                else:
                    st.progress(similarity_percent / 100)
                    st.warning(f"⚠️ **Not a dupe.** {similarity_percent:.1f}% similarity (below {threshold}% threshold)")

st.markdown("---")
st.caption("Fashion Dupe Detection Demo | Built with Streamlit")
