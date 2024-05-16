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

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()))
    image = cv2.imdecode(file_bytes, 1)

    imgRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    st.write("")
    st.write("Detecting...")
    result = model(imgRGB, size=600)
  
    detect_class = result.pandas().xyxy[0]
  
    st.code(detect_class[['name', 'xmin','ymin', 'xmax', 'ymax']])
  
    # Count each class
    class_counts = detect_class['name'].value_counts().to_dict()
    st.write("Class counts:")
    
    for class_name, count in class_counts.items():
        st.write(f"- **{class_name.capitalize()}**: {count}")

    
  
    outputpath = 'output.jpg'
  
    result.render()  # render bbox in image
    for im in result.ims:
        im_base64 = Image.fromarray(im)
        im_base64.save(outputpath)
        img_ = Image.open(outputpath)
        st.image(img_, caption='Model Prediction(s)')
    
    for idx, row in detect_class.iterrows():
        xmin, ymin, xmax, ymax, name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name']
        cropped_image = imgRGB[ymin:ymax, xmin:xmax]
        st.image(cropped_image, caption=f"{name.capitalize()}" ,width=300)

  


