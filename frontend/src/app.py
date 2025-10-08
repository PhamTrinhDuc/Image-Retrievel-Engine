import streamlit as st
import requests
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Image Retrieval System",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv(f"{os.getenv('BACKEND_HOST')}:{os.getenv('BACKEND_PORT')}", "http://localhost:8000")

class ImageRetrievalApp:
    def __init__(self):
        self.api_url = API_BASE_URL
    
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def search_similar_images(self, image_file, top_k=5, extractor_type="resnet"):
        """Search for similar images"""
        try:
            files = {"file": ("image.jpg", image_file, "image/jpeg")}
            data = {
                "top_k": top_k,
                "extractor_type": extractor_type
            }
            
            response = requests.post(
                f"{self.api_url}/search/upload",
                files=files,
                data=data,
                timeout=30
            )
            
            return response.status_code == 200, response.json()
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_available_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json().get("available_extractors", ["resnet"])
            return ["resnet"]
        except:
            return ["resnet"]

def main():
    st.title("🔍 Image Retrieval System")
    st.markdown("Upload an image to find similar images in the database")
    
    app = ImageRetrievalApp()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API Status
        st.subheader("🔗 API Status")
        is_healthy, health_data = app.check_api_health()
        
        if is_healthy:
            st.success("✅ API is running")
            if "models_loaded" in health_data:
                for model, loaded in health_data["models_loaded"].items():
                    status = "✅" if loaded else "❌"
                    st.text(f"{status} {model}")
        else:
            st.error("❌ API is not available")
            st.text("Make sure to start the API server:")
            st.code("cd source/api && python run_server.py")
            return
        
        # Model Selection
        st.subheader("🤖 Model Settings")
        available_models = app.get_available_models()
        selected_model = st.selectbox(
            "Choose extractor model:",
            available_models,
            index=0
        )
        
        # Search Parameters
        top_k = st.slider("Number of results:", 1, 20, 5)
        
        st.subheader("ℹ️ Available Models")
        model_info = {
            "resnet": "ResNet-50 pretrained",
            "vgg": "VGG-16 features", 
            "vit": "Vision Transformer",
            "dinov2": "DINOv2 features",
        }
        
        for model in available_models:
            if model in model_info:
                st.text(f"• {model}: {model_info[model]}")
    
    # Main content
    st.markdown("---")
    
    # Upload section
    st.header("📤 Upload Image")
    
    col_upload, col_preview = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to search for similar ones"
        )
    
    # Preview and search section
    if uploaded_file is not None:
        with col_preview:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)
        
        # Search button (full width)
        st.markdown("###")
        if st.button("🔍 Search Similar Images", type="primary", use_container_width=True):
            with st.spinner(f"Searching with {selected_model} model..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Search for similar images
                success, result = app.search_similar_images(
                    uploaded_file, 
                    top_k=top_k, 
                    extractor_type=selected_model
                )
                
                if success and result.get("success"):
                    st.session_state.search_results = result
                    st.session_state.query_image = image
                else:
                    st.error(f"Search failed: {result.get('message', 'Unknown error')}")
    
    # Results section
    st.markdown("---")
    st.header("🎯 Search Results")
    
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
        results = st.session_state.search_results
        
        # Display search info in columns
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.success(f"✅ {results.get('message', 'Search completed')}")
        with info_col2:
            st.info(f"⏱️ Query time: {results.get('query_time', 0):.3f} seconds")
        
        # Display results
        similar_images = results.get("results", [])
        
        if similar_images:
            st.subheader(f"Found {len(similar_images)} similar images:")
            
            # Display results in a grid
            for i in range(0, len(similar_images), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(similar_images):
                        result = similar_images[i + j]
                        with col:
                            with st.container():
                                st.markdown(f"**Result {i+j+1}**")
                                st.metric("Similarity Score", f"{result.get('similarity_score', 0):.3f}")
                                st.text(f"ID: {result.get('image_id', 'unknown')}")
                                
                                # Show metadata if available
                                if result.get("metadata"):
                                    with st.expander("Metadata"):
                                        st.json(result.get("metadata", {}))
        else:
            st.warning("No similar images found")
    else:
        st.info("Upload an image and click search to see results here")
    
    # Footer
    st.markdown("---")
    st.subheader("🔧 System Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏥 Check API Health", use_container_width=True):
            is_healthy, health_data = app.check_api_health()
            if is_healthy:
                st.success("API is healthy")
                with st.expander("Health Details"):
                    st.json(health_data)
            else:
                st.error("API is down")
                with st.expander("Error Details"):
                    st.json(health_data)
    
    with col2:
        if st.button("🤖 List Models", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/models")
                if response.status_code == 200:
                    data = response.json()
                    st.success("Models loaded")
                    with st.expander("Available Models"):
                        st.json(data)
                else:
                    st.error("Failed to get models")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col3:
        if st.button("🗄️ Database Info", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/vdb")
                if response.status_code == 200:
                    data = response.json()
                    st.success("Database connected")
                    with st.expander("Database Details"):
                        st.json(data)
                else:
                    st.error("Failed to get DB info")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # App info
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Image Retrieval System | Built with Streamlit + FastAPI
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()