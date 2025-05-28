import streamlit as st
import cv2
import numpy as np
import time
from imutils.video import VideoStream
import imutils

weights_path='yolov3.weights'
config_path='yolov3.cfg'
labels_path='coco.names'


# Load the class labels
with open(labels_path, 'r') as f:
    CLASSES=f.read().strip().split('\n')

COLORS=np.random.uniform(0, 255, size=(len(CLASSES),3))

# Load YOLO objecr detector
net=cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln=net.getLayerNames()
ln=[ln[i-1] for i in net.getUnconnectedOutLayers().flatten()]

# Streamlit UI
st.title("Real-time Object Detection")
run=st.checkbox("Start Webcam")
FRAME_WINDOW=st.image([])
confidence_threshold=0.3

if run:
    vs=VideoStream(src=0).start()
    time.sleep(2.0)

    while run:
        frame=vs.read()
        frame=imutils.resize(frame, width=500)
        (H,W)=frame.shape[:2]

        blob=cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs=net.forward(ln)

        boxes=[]
        confidences=[]
        class_ids=[]

        for output in layer_outputs:
            for detection in output:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]

                if confidence>confidence_threshold:
                    box=detection[0:4]*np.array([W,H, W,H])
                    (centerX, centerY, width, height)=box.astype("int")
                    x=int(centerX-width/2)
                    y=int(centerY-height/2)

                    boxes.append([x,y,int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs=cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.3)

        if len(idxs)>0:
            for i in idxs.flatten():
                (x,y)=(boxes[i][0], boxes[i][1])
                (w,h)=(boxes[i][2], boxes[i][3])
                color=COLORS[class_ids[i]]
                label=f"{CLASSES[class_ids[i]]}: {confidences[i]*100:.2f}%"

                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    vs.stop()
else:
    st.write("Webcam feed stopped.")

        

