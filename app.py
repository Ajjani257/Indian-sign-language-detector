# app.py

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ISL Gesture Recognition",
    page_icon="👋",
    layout="wide"
)

# --- UI AND STYLING ---
st.title("Indian Sign Language (ISL) Hand Gesture Recognition 🖐️")
st.write("This app uses your webcam to detect and classify ISL hand gestures in real-time. Make sure your hand is clearly visible.")

# --- MODEL AND MEDIAPIPE LOADING ---
# Caching the resources to prevent reloading on every interaction
@st.cache_resource
def load_model_and_encoder():
    """Loads the pre-trained model and label encoder."""
    try:
        model = joblib.load("isl_landmark_model_2hands.pkl")
        label_encoder = joblib.load("label_encoder_2hands.pkl")
        return model, label_encoder
    except FileNotFoundError:
        st.error("Model or Label Encoder not found.")
        st.info("Please ensure 'isl_landmark_model_2hands.pkl' and 'label_encoder_2hands.pkl' are in the same directory as app.py.")
        return None, None

@st.cache_resource
def initialize_mediapipe():
    """Initializes and returns MediaPipe Hands solution."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_draw, mp_hands

model, label_encoder = load_model_and_encoder()
hands, mp_draw, mp_hands = initialize_mediapipe()

# --- WEBCAM AND PREDICTION LOGIC ---
if model and label_encoder:
    # Use session state to control the webcam feed
    if 'run' not in st.session_state:
        st.session_state.run = False

    def start_detection():
        st.session_state.run = True

    def stop_detection():
        st.session_state.run = False

    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        st.button("Start ▶️", on_click=start_detection)
    with col2:
        st.button("Stop ⏹️", on_click=stop_detection)

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if st.session_state.run:
        st.info("Webcam is active. Position your hand to start detection.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                st.session_state.run = False
                break

            # Process the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            landmarks_flattened = []
            if results.multi_hand_landmarks:
                # Extract landmarks for up to 2 hands
                for hand_landmarks in results.multi_hand_landmarks[:2]:
                    for lm in hand_landmarks.landmark:
                        landmarks_flattened.extend([lm.x, lm.y, lm.z])
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Pad to the required feature size (126 for 2 hands)
                while len(landmarks_flattened) < 126:
                    landmarks_flattened.extend([0.0, 0.0, 0.0])
                landmarks_flattened = landmarks_flattened[:126]

                # Predict
                data_np = np.array(landmarks_flattened).reshape(1, -1)
                prediction = model.predict(data_np)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                probability = np.max(model.predict_proba(data_np))
                
                # Display prediction
                text = f"Prediction: {predicted_label} ({probability * 100:.1f}%)"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            FRAME_WINDOW.image(frame, channels="BGR")

            if not st.session_state.run:
                break
    else:
        st.warning("Webcam is off. Click 'Start' to begin gesture detection.")
    
    cap.release()
