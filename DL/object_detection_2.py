import streamlit as st
import cv2
import numpy as np
import time
from imutils.video import VideoStream
import imutils

st.title("Real-time Object Detection using MobileNet SSD")
run=st.checkbox("Start Webcam")
FRAME_WINDOW=st.image([])

prototxt_path='MobileNetSSD_deploy.prototxt.txt'
model_path='MobileNetSSD_deploy.caffemodel'
confidence_threshold=0.2

CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "mobilephone"]

COLORS=np.random.uniform(0, 255, size=(len(CLASSES),3))

net=cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

if run:
    vs=VideoStream(src=0).start()
    time.sleep(2.0)

    while run:
        frame=vs.read()
        frame=imutils.resize(frame, width=500)
        (H,W)=frame.shape[:2]

        blob=cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        detections=net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence > confidence_threshold:
                idx=int(detections[0,0,i,1])
                box=detections[0, 0, i, 3:7]*np.array([W,H, W,H])

                (startX, startY, endX, endY)=box.astype("int")
                label=f"{CLASSES[idx]}: {confidence*100:.2f}%"

                cv2.rectangle(frame, (startX,startY), (endX, endY), COLORS[idx], 2)
                y=startY-15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    vs.stop()
else:
    st.write("Webcam feed stopped.")
