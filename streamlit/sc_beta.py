import cv2 
import streamlit as st
import numpy as np 
from predict_camera_1 import img_live
import pickle
from PIL import Image

# load model
with open('RF_model.pkl', 'rb') as f:
    rf = pickle.load(f)
    
st.markdown("<h1 style='text-align: center; color: white;'>SIGN LANGUAGE</h1>", unsafe_allow_html=True)
option = st.selectbox('Do you want to predict with your camera?', ["Home", 'Yes', 'No'])

cam = cv2.VideoCapture(0)
if "Home" in option:
    st.image("lenguaje_signos_2.jpg", width=705)
elif 'Yes' in option:
    run="Yes"
    FRAME_WINDOW = st.image([])
    while run:
        ret, frame = cam.read()
        data = img_live(frame)
        data = np.array(data)
        y_pred = rf.predict(data.reshape(-1,63))
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 100)
        fontScale = 3
        color = (200, 175, 175)
        thickness = 5
        letter = str(y_pred[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.putText(frame, letter, position, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW.image(frame)
elif 'No' in option: 
    st.subheader("Maybe next time!!!")
    cam.release()
    cv2.destroyAllWindows()


