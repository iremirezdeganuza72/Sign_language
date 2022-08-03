# main #

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2 
import numpy as np 
import pickle
import av
import threading
import os
from predict_camera_1 import img_live

separator = os.path.sep
path_act = os.path.dirname(os.path.abspath(__file__))
dir = separator.join(path_act.split(separator)[:-1])

with open('RF_model.pkl', 'rb') as f:
    rf = pickle.load(f)
 

st.title("Sign language")
select = st.sidebar.selectbox("What do you want?",["Streaming", "Upload Image"])

if select == "Streaming":
    class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model_lock = threading.Lock()

            def recv(self, frame):
                while True:
                    ret, frame = cap.read()
                    data = img_live(frame)
                    data = np.array(data)
                    y_pred = rf.predict(data.reshape(-1,63))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    position = (50, 100)
                    fontScale = 3
                    color = (200, 175, 175)
                    thickness = 5
                    letter = str(y_pred[0])
                    frame = cv2.putText(frame, letter, position, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
                    return cv2.imshow('Signals', frame)

            if cv2.waitKey(1) == ord('q'):
                cap.release

    webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )
elif select == "Upload Image":
    st.subheader("Upload Image")
    frame = st.file_uploader("Upload Image", type=["jpg", "png"])
    if frame is not None:
        file_bytes = np.asarray(bytearray(frame.read())).astype(np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        data = img_live(image)
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1,63))
        st.image(frame)
        st.subheader(f'The sign is a {y_pred[0]}')
    else:
        st.write("No Image Selected")

        
        
        
        




