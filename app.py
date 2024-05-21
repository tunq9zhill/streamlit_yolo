import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect

st.title("🌊 Under the sea detection")

st.write("Upload your Image...")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)

def detect_objects(frame):
    # Convert frame to a suitable format for YOLOv5
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(img, size=640)  # Reduce size if needed
    return results
def main():
    run = st.checkbox('Run')
    stframe = st.empty()
    video_capture = cv2.VideoCapture(0)

    while run:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Failed to capture image")
            break
        
        results = detect_objects(frame)
        results.render()  # Updates results.imgs with boxes and labels
        
        # Convert the frame with detections to an image format for streamlit
        img = Image.fromarray(cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB))
        stframe.image(np.array(img), channels="RGB")
    
    video_capture.release()

if __name__ == "__main__":
    main()
  


