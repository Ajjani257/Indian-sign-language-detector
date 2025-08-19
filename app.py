import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from PIL import Image
import pickle
import os
from io import BytesIO
import base64

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="🧠 ISL Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3em;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #2E86AB;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2em;
    }
    .info-box {
        background-color: #e8f4f8;
        border: 1px solid #2E86AB;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.label_encoder = None

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        return mp_hands, hands, mp_draw, True
    except Exception as e:
        st.error(f"Error loading MediaPipe: {e}")
        return None, None, None, False

# Create a simple model for demo purposes if files don't exist
def create_demo_model():
    """Create a simple demo model and label encoder for demonstration"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    # Create dummy data for ISL classes (A-Z, 0-9)
    classes = [chr(i) for i in range(ord('A'), ord('Z')+1)] + [str(i) for i in range(10)]
    
    # Create label encoder
    le = LabelEncoder()
    le.fit(classes)
    
    # Create a simple model (this is just for demo - not actually trained)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    
    # Create dummy training data
    X_dummy = np.random.random((len(classes) * 10, 126))  # 126 features for 2 hands
    y_dummy = np.repeat(le.transform(classes), 10)
    
    model.fit(X_dummy, y_dummy)
    
    return model, le

# Load or create model
def load_or_create_model():
    try:
        # Try to load existing model files
        if os.path.exists("isl_landmark_model_2hands.pkl") and os.path.exists("label_encoder_2hands.pkl"):
            model = joblib.load("isl_landmark_model_2hands.pkl")
            label_encoder = joblib.load("label_encoder_2hands.pkl")
            return model, label_encoder, "loaded"
        else:
            # Create demo model
            model, label_encoder = create_demo_model()
            return model, label_encoder, "demo"
    except Exception as e:
        st.error(f"Error with model: {e}")
        return None, None, "error"

# Extract landmarks from image
def extract_landmarks(image, hands):
    """Extract hand landmarks from an image"""
    try:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
        else:
            rgb_image = np.array(image)
        
        results = hands.process(rgb_image)
        landmarks_flattened = []
        
        if results.multi_hand_landmarks:
            # Extract landmarks for up to 2 hands
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    landmarks_flattened.extend([lm.x, lm.y, lm.z])
            
            # Pad to 126 (21 landmarks × 3 coords × 2 hands)
            while len(landmarks_flattened) < 126:
                landmarks_flattened.extend([0.0, 0.0, 0.0])
            
            landmarks_flattened = landmarks_flattened[:126]
            return np.array(landmarks_flattened).reshape(1, -1), results.multi_hand_landmarks
        
        return None, None
    except Exception as e:
        st.error(f"Error extracting landmarks: {e}")
        return None, None

# Draw landmarks on image
def draw_landmarks_on_image(image, hand_landmarks_list, mp_hands, mp_draw):
    """Draw hand landmarks on the image"""
    try:
        annotated_image = np.array(image).copy()
        
        for hand_landmarks in hand_landmarks_list:
            mp_draw.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        return annotated_image
    except Exception as e:
        st.error(f"Error drawing landmarks: {e}")
        return np.array(image)

# Predict sign language
def predict_sign(landmarks, model, label_encoder):
    """Predict the sign language character"""
    try:
        if landmarks is not None and model is not None:
            prediction = model.predict(landmarks)
            probabilities = model.predict_proba(landmarks)
            confidence = np.max(probabilities) * 100
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            return predicted_label, confidence
        return None, 0
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, 0

# File uploader for model files
def model_uploader():
    """Allow users to upload their own model files"""
    st.sidebar.subheader("🔧 Upload Model Files (Optional)")
    
    model_file = st.sidebar.file_uploader(
        "Upload Model (.pkl)", 
        type=['pkl'],
        help="Upload your trained ISL model file"
    )
    
    encoder_file = st.sidebar.file_uploader(
        "Upload Label Encoder (.pkl)", 
        type=['pkl'],
        help="Upload your label encoder file"
    )
    
    if model_file and encoder_file:
        if st.sidebar.button("🔄 Load Uploaded Models"):
            try:
                # Load uploaded files
                model = joblib.load(model_file)
                label_encoder = joblib.load(encoder_file)
                
                # Save to session state
                st.session_state.model = model
                st.session_state.label_encoder = label_encoder
                st.session_state.model_loaded = True
                
                st.sidebar.success("✅ Models loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error loading models: {e}")

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Indian Sign Language Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time ISL recognition using MediaPipe & MLP Classifier</p>', unsafe_allow_html=True)
    
    # Load MediaPipe
    mp_hands, hands, mp_draw, mp_success = load_mediapipe()
    
    if not mp_success:
        st.error("❌ Failed to load MediaPipe. Please check your installation.")
        st.stop()
    
    # Load or create model
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI models..."):
            model, label_encoder, model_status = load_or_create_model()
            
            if model is not None and label_encoder is not None:
                st.session_state.model = model
                st.session_state.label_encoder = label_encoder
                st.session_state.model_loaded = True
                
                if model_status == "demo":
                    st.warning("⚠️ Using demo model. Upload your trained model files for better accuracy.")
                elif model_status == "loaded":
                    st.success("✅ Trained model loaded successfully!")
            else:
                st.error("❌ Failed to load or create model.")
                st.stop()
    
    # Sidebar
    st.sidebar.header("🎯 Detection Options")
    
    # Model uploader
    model_uploader()
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Detection Mode:",
        ["📷 Upload Image", "📊 Batch Processing", "📖 About"]
    )
    
    # Display supported signs
    with st.sidebar.expander("🔤 Supported Signs"):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.write("**Alphabets:**")
            st.write("A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z")
        with col2:
            st.write("**Digits:**")
            st.write("0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
        st.write("**Total Classes:** 36")
    
    # Main content area
    if mode == "📷 Upload Image":
        st.header("📷 Upload Image for ISL Detection")
        
        st.markdown("""
        <div class="info-box">
        <strong>📝 Instructions:</strong><br>
        1. Upload a clear image showing hand gesture(s)<br>
        2. Ensure good lighting and hand visibility<br>
        3. The system can detect up to 2 hands simultaneously<br>
        4. Supported formats: PNG, JPG, JPEG
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing ISL hand gestures"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner("🔍 Analyzing hand gestures..."):
                landmarks, hand_landmarks_list = extract_landmarks(image, hands)
                
                if landmarks is not None:
                    predicted_label, confidence = predict_sign(
                        landmarks, st.session_state.model, st.session_state.label_encoder
                    )
                    
                    # Draw landmarks
                    annotated_image = draw_landmarks_on_image(
                        image, hand_landmarks_list, mp_hands, mp_draw
                    )
                    
                    with col2:
                        st.subheader("🎯 Detection Result")
                        st.image(annotated_image, use_column_width=True)
                    
                    # Display prediction
                    if confidence > 80:
                        confidence_class = "confidence-high"
                        confidence_emoji = "🟢"
                    elif confidence > 60:
                        confidence_class = "confidence-medium"
                        confidence_emoji = "🟡"
                    else:
                        confidence_class = "confidence-low"
                        confidence_emoji = "🔴"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🔮 Prediction Results</h3>
                        <p><strong>Detected Sign:</strong> 
                        <span style="font-size: 3em; color: #2E86AB; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">{predicted_label}</span></p>
                        <p><strong>Confidence:</strong> {confidence_emoji} <span class="{confidence_class}">{confidence:.1f}%</span></p>
                        <p><strong>Hands Detected:</strong> {len(hand_landmarks_list)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional info
                    if confidence < 70:
                        st.info("💡 **Tip:** For better accuracy, try uploading an image with clearer hand positioning and better lighting.")
                    
                else:
                    st.warning("❌ No hands detected in the image. Please try with a clearer image showing hand gestures.")
                    st.markdown("""
                    **Suggestions:**
                    - 🔆 Ensure good lighting
                    - 🖐️ Make sure hands are clearly visible
                    - 📏 Keep hands at appropriate distance from camera
                    - 🎯 Position hands in the center of the image
                    """)
    
    elif mode == "📊 Batch Processing":
        st.header("📊 Batch Image Processing")
        
        st.markdown("""
        <div class="info-box">
        <strong>🚀 Batch Processing Features:</strong><br>
        • Process multiple images simultaneously<br>
        • Export results as CSV<br>
        • Progress tracking<br>
        • Detailed analysis report
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose multiple images...", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                process_btn = st.button("🚀 Process All Images", type="primary")
            
            with col2:
                show_images = st.checkbox("🖼️ Show processed images")
            
            if process_btn:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    try:
                        image = Image.open(uploaded_file)
                        landmarks, hand_landmarks_list = extract_landmarks(image, hands)
                        
                        if landmarks is not None:
                            predicted_label, confidence = predict_sign(
                                landmarks, st.session_state.model, st.session_state.label_encoder
                            )
                            results.append({
                                'Filename': uploaded_file.name,
                                'Predicted_Sign': predicted_label,
                                'Confidence_%': round(confidence, 2),
                                'Hands_Detected': len(hand_landmarks_list) if hand_landmarks_list else 0,
                                'Status': 'Success'
                            })
                            
                            if show_images:
                                with st.expander(f"📸 {uploaded_file.name} - Prediction: {predicted_label}"):
                                    col_img1, col_img2 = st.columns(2)
                                    with col_img1:
                                        st.image(image, caption="Original", use_column_width=True)
                                    with col_img2:
                                        annotated_image = draw_landmarks_on_image(
                                            image, hand_landmarks_list, mp_hands, mp_draw
                                        )
                                        st.image(annotated_image, caption="Processed", use_column_width=True)
                        else:
                            results.append({
                                'Filename': uploaded_file.name,
                                'Predicted_Sign': 'N/A',
                                'Confidence_%': 0.0,
                                'Hands_Detected': 0,
                                'Status': 'No Hand Detected'
                            })
                    
                    except Exception as e:
                        results.append({
                            'Filename': uploaded_file.name,
                            'Predicted_Sign': 'Error',
                            'Confidence_%': 0.0,
                            'Hands_Detected': 0,
                            'Status': f'Error: {str(e)[:50]}'
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Display results
                if results:
                    st.success(f"🎉 Processed {len(results)} images successfully!")
                    
                    # Summary statistics
                    df_results = pd.DataFrame(results)
                    successful_predictions = len(df_results[df_results['Status'] == 'Success'])
                    avg_confidence = df_results[df_results['Status'] == 'Success']['Confidence_%'].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Images", len(results))
                    with col2:
                        st.metric("Successful Detections", successful_predictions)
                    with col3:
                        st.metric("Success Rate", f"{(successful_predictions/len(results)*100):.1f}%")
                    with col4:
                        if not pd.isna(avg_confidence):
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        else:
                            st.metric("Avg Confidence", "N/A")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="isl_detection_results.csv",
                        mime="text/csv"
                    )
    
    elif mode == "📖 About":
        st.header("📖 About ISL Detector")
        
        # Project overview
        st.markdown("""
        <div class="feature-card">
        <h3>🎯 Project Overview</h3>
        <p>This application uses advanced computer vision and machine learning to recognize 
        Indian Sign Language (ISL) gestures in real-time. Built with MediaPipe for hand 
        landmark detection and Scikit-learn's MLP Classifier for gesture classification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h4>✨ Key Features</h4>
            <ul>
            <li>🖐️ Dual-hand detection support</li>
            <li>📷 Single image processing</li>
            <li>📊 Batch processing capability</li>
            <li>🎯 36 ISL signs (A-Z, 0-9)</li>
            <li>📈 Confidence scoring</li>
            <li>💾 Export results to CSV</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h4>🔧 Technical Stack</h4>
            <ul>
            <li>🧠 <strong>MediaPipe:</strong> Hand landmark detection</li>
            <li>🤖 <strong>Scikit-learn:</strong> MLP Classifier</li>
            <li>🖼️ <strong>OpenCV:</strong> Image processing</li>
            <li>🚀 <strong>Streamlit:</strong> Web interface</li>
            <li>🐍 <strong>Python:</strong> Core development</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # How it works
        st.markdown("""
        <div class="feature-card">
        <h3>⚙️ How It Works</h3>
        <ol>
        <li><strong>Hand Detection:</strong> MediaPipe identifies hand regions and extracts 21 3D landmarks per hand</li>
        <li><strong>Feature Extraction:</strong> Landmark coordinates are flattened into a 126-dimensional feature vector</li>
        <li><strong>Classification:</strong> MLP Neural Network classifies the gesture into one of 36 ISL signs</li>
        <li><strong>Prediction:</strong> System outputs the predicted sign with confidence score</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Usage tips
        st.markdown("""
        <div class="feature-card">
        <h3>💡 Usage Tips</h3>
        <ul>
        <li>📸 <strong>Image Quality:</strong> Use clear, well-lit images for best results</li>
        <li>🖐️ <strong>Hand Position:</strong> Keep hands centered and clearly visible</li>
        <li>📏 <strong>Distance:</strong> Maintain appropriate distance from camera</li>
        <li>🎯 <strong>Background:</strong> Use contrasting background for better detection</li>
        <li>🔄 <strong>Model:</strong> Upload your own trained model for better accuracy</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Model info
        if st.session_state.model_loaded:
            try:
                model_info = {
                    "Model Type": "MLP Classifier",
                    "Input Features": "126 (21 landmarks × 3 coordinates × 2 hands)",
                    "Output Classes": len(st.session_state.label_encoder.classes_),
                    "Supported Signs": ", ".join(sorted(st.session_state.label_encoder.classes_))
                }
                
                st.markdown("### 🔍 Current Model Information")
                for key, value in model_info.items():
                    if key == "Supported Signs":
                        st.write(f"**{key}:** {value}")
                    else:
                        st.write(f"**{key}:** {value}")
            except Exception as e:
                st.write("Model information not available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤟 <strong>ISL Detector</strong> - Bridging Communication Gaps with AI</p>
        <p>Built with ❤️ using MediaPipe, Scikit-learn & Streamlit</p>
        <p><em>Making sign language recognition accessible to everyone</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
