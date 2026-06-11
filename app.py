# app.py
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import tempfile
import os

st.set_page_config(page_title="Violence Detection App", layout="wide")

st.title("🎥 Violence Detection in Videos & Live Camera")
st.write("""
Upload a video or use your webcam and the AI model will predict if violence is detected.  
The model uses deep learning (MobileNetV2 + LSTM) with TensorFlow.
""")

# -------------------------------
# Load LSTM model
# -------------------------------
@st.cache_data(show_spinner=False)
def load_model():
    with st.spinner("Loading model..."):
        model = tf.keras.models.load_model("violencevideo_model.h5", compile=False)
    return model

model = load_model()
st.success("✅ LSTM model loaded successfully!")

# -------------------------------
# Load feature extractor (MobileNetV2 without pooling)
# -------------------------------
@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)  # output: (7,7,1280)
    return feature_extractor

feature_extractor = load_feature_extractor()

# -------------------------------
# Extract frames from video
# -------------------------------
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    idx = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i == frame_indices[idx]:
            frame = cv2.resize(frame, (224,224))
            frame = frame / 255.0
            frames.append(frame)
            idx += 1
            if idx >= len(frame_indices):
                break
    cap.release()
    return np.array(frames)

# -------------------------------
# Convert frames to CNN features
# -------------------------------
def frames_to_features(frames):
    features = feature_extractor.predict(frames, verbose=0)  # shape: (num_frames,7,7,1280)
    return np.expand_dims(features, axis=0)  # shape: (1,num_frames,7,7,1280)

# -------------------------------
# Video upload detection
# -------------------------------
st.subheader("1️⃣ Upload a video file")
uploaded_file = st.file_uploader("Upload MP4/AVI video", type=["mp4","avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    st.write("Processing video, please wait...")

    try:
        frames = extract_frames(video_path)
        frames_input = frames_to_features(frames)
        prediction = model.predict(frames_input)
        violence_prob = float(prediction[0][0])

        if violence_prob > 0.5:
            st.error(f"⚠️ Violence detected! Confidence: {violence_prob:.2f}")
        else:
            st.success(f"✅ No violence detected. Confidence: {1-violence_prob:.2f}")

    finally:
        try:
            os.remove(video_path)
        except PermissionError:
            st.warning("⚠️ Could not delete temp video file immediately. Try again later.")

# -------------------------------
# Live webcam detection
# -------------------------------
st.subheader("2️⃣ Real-time webcam detection")
start_cam = st.checkbox("Start Webcam")

if start_cam:
    st.write("Live webcam running. Predictions update every 16 frames.")
    cap = cv2.VideoCapture(0)
    num_frames = 16
    frame_seq = []
    webcam_placeholder = st.empty()  # placeholder to update frames

    stop_cam = False
    while not stop_cam:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break

        # Display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        webcam_placeholder.image(frame_rgb, channels="RGB")

        # Prepare frame
        f = cv2.resize(frame, (224,224)) / 255.0
        frame_seq.append(f)

        # Predict every 16 frames
        if len(frame_seq) == num_frames:
            frames_array = np.array(frame_seq)
            frames_input = frames_to_features(frames_array)
            prediction = model.predict(frames_input)
            violence_prob = float(prediction[0][0])

            if violence_prob > 0.5:
                st.warning(f"⚠️ Violence detected! Confidence: {violence_prob:.2f}")
            else:
                st.success(f"✅ No violence detected. Confidence: {1-violence_prob:.2f}")

            # Sliding window
            frame_seq.pop(0)

        # Stop webcam if checkbox is unchecked
        if not st.session_state.get("run_webcam", True):
            stop_cam = True

    cap.release()