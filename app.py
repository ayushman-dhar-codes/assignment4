from taipy.gui import Gui, State
import cv2
import numpy as np
import os
import urllib.request

def download_models():
    prototxt_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel"
    
    if not os.path.exists("MobileNetSSD_deploy.prototxt"):
        print("Downloading architecture file...")
        urllib.request.urlretrieve(prototxt_url, "MobileNetSSD_deploy.prototxt")
            
    if not os.path.exists("MobileNetSSD_deploy.caffemodel"):
        print("Downloading AI weights file...")
        urllib.request.urlretrieve(model_url, "MobileNetSSD_deploy.caffemodel")

download_models()

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

image_path = None
processed_image_bytes = None
min_confidence = 0.5
analytics_text = "Please upload an image from the sidebar to begin."

def process_image(state: State):
    if not state.image_path:
        return


    image_cv2 = cv2.imread(state.image_path)
    if image_cv2 is None:
        state.analytics_text = "Error reading image."
        return

    (h, w) = image_cv2.shape[:2]
    

    blob = cv2.dnn.blobFromImage(image_cv2, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []


    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > state.min_confidence:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASS_NAMES[idx]}: {confidence * 100:.2f}%"
            detected_objects.append(CLASS_NAMES[idx])
            
            cv2.rectangle(image_cv2, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image_cv2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    _, buffer = cv2.imencode('.jpg', image_cv2)
    state.processed_image_bytes = buffer.tobytes()


    if len(detected_objects) > 0:
        unique_objects = set(detected_objects)
        lines = [f"**Successfully identified {len(detected_objects)} objects.**"]
        for obj in unique_objects:
            count = detected_objects.count(obj)
            lines.append(f"* **{obj.capitalize()}**: {count} detected")
        state.analytics_text = "\n".join(lines)
    else:
        state.analytics_text = "⚠️ No objects detected with the current confidence threshold. Try lowering the slider."

def on_change(state: State, var_name: str, var_value):
    if var_name in ["image_path", "min_confidence"]:
        if state.image_path:
            process_image(state)

page = """
# Automated Object Detection & Pattern Analyzer

Upload an image to identify objects using a MobileNet-SSD Machine Learning model.

<|layout|columns=1 3|
<|
### Control Panel
<|{image_path}|file_selector|label=Upload Image|extensions=.jpg,.png,.jpeg|>

**Confidence Threshold:** <|{min_confidence}|>
<|{min_confidence}|slider|min=0.0|max=1.0|step=0.05|>
|>

<|
<|layout|columns=1 1|
<|
### Original Image
<|{image_path}|image|width=100%|>
|>
<|
### Processed Image
<|{processed_image_bytes}|image|width=100%|>
|>
|>

### Identification Analytics
<|{analytics_text}|text|mode=markdown|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="Object Detection GUI", use_reloader=True)