# Automated Object Detection & Pattern Analyzer

A real-time, web-based Graphical User Interface (GUI) for identifying and analyzing objects within images. This application leverages a pre-trained Deep Neural Network to perform pattern recognition and object detection, demonstrating practical applications of computer vision and machine learning.

## 🎯 Problem Statement
Manual monitoring of visual data (such as urban traffic, security feeds, or media) is highly inefficient. There is a critical need for automated systems capable of instantly identifying, classifying, and counting multiple entities within a given frame. This project solves that problem by providing a user-friendly interface that overlays spatial bounding boxes and generates detection analytics in real-time.

## ✨ Key Features
* **Competitive GUI:** Built with Streamlit, featuring a modern, responsive split-screen layout for instant visual comparison.
* **Machine Learning Integration:** Utilizes a Caffe-based MobileNet Single Shot Detector (SSD) for high-speed, accurate object detection.
* **Automated Dependency Management:** The script automatically fetches and securely downloads the required, OpenCV-optimized `.prototxt` and `.caffemodel` files if they are not present in the directory.
* **Dynamic Thresholding:** Users can adjust the AI's confidence threshold on the fly via a sidebar slider to filter out false positives.
* **Data Analytics:** Automatically counts and categorizes identified patterns, displaying the data in a clean analytics dashboard.

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Frontend/GUI:** Streamlit
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy
* **Image Handling:** Pillow (`PIL`)

## 📦 Installation & Setup

1. **Clone or Download the Repository**
   Ensure all files are within the same project directory.

2. **Install Required Libraries**
   Open your terminal and run the following command to install the necessary dependencies:
   ```bash
   pip install streamlit opencv-python numpy pillow