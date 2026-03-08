import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

st.set_page_config(page_title="Object Detection GUI", layout="wide")
st.title("Automated Object Detection & Pattern Analyzer")
st.markdown("Upload an image to identify objects using a MobileNet-SSD Machine Learning model.")

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

@st.cache_resource
def download_models():

    prototxt_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel"
    
    if not os.path.exists("MobileNetSSD_deploy.prototxt"):
        with st.spinner("Downloading architecture file..."):
            urllib.request.urlretrieve(prototxt_url, "MobileNetSSD_deploy.prototxt")
            
    if not os.path.exists("MobileNetSSD_deploy.caffemodel"):
        with st.spinner("Downloading AI weights file (this might take a minute)..."):
            urllib.request.urlretrieve(model_url, "MobileNetSSD_deploy.caffemodel")

download_models()

@st.cache_resource
def load_model():

    net = cv2.dnn.readNetFromCaffe(
        "MobileNetSSD_deploy.prototxt", 
        "MobileNetSSD_deploy.caffemodel"
    )
    return net

net = load_model()

st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Image (jpg, png)", type=["jpg", "png", "jpeg"])
min_confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    (h, w) = image_cv2.shape[:2]


    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv2, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    with st.spinner('Running Machine Learning Model...'):
        detections = net.forward()

    detected_objects = []

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASS_NAMES[idx]}: {confidence * 100:.2f}%"
            detected_objects.append(CLASS_NAMES[idx])
            
            cv2.rectangle(image_cv2, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image_cv2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Processed Image (Object Identification)")
        st.image(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB), use_container_width=True)


    st.subheader("Identification Analytics")
    if len(detected_objects) > 0:
        st.success(f"Successfully identified {len(detected_objects)} objects.")
        
        unique_objects = set(detected_objects)
        for obj in unique_objects:
            count = detected_objects.count(obj)
            st.write(f"- **{obj.capitalize()}**: {count} detected")
    else:
        st.warning("No objects detected with the current confidence threshold. Try lowering the slider.")
else:
    st.info("Please upload an image from the sidebar to begin.")