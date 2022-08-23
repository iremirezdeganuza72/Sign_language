from turtle import width
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
option = st.sidebar.selectbox ( 'Choose: Home, predict with your camera or predict uploading an image.' , ["Home", 'Predict with my camera', 'Upload an image'] )

if option=="Home":
    option_2= st.sidebar.selectbox( 'What kind of let1ter do you want to predict?', [ "SUMMARY", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "K", "Y", "Z" ] )

    if option_2:
        st.image(f"{option_2}.jpg",  width=705)
    

elif option=='Predict with my camera':
    st.image([])
    st.subheader("")
    cam = cv2.VideoCapture(0)
    run=True
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

elif option == 'Upload an image':
    frame = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "gif", "tiff", "bmp"])
    if frame is not None:
        file_bytes = np.asarray(bytearray(frame.read())).astype(np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        data = img_live(image)
        data = np.array(data)
        y_pred = rf.predict(data.reshape(-1,63))
        st.image(frame)
        st.subheader(f'The sign is a {y_pred[0]}')
    else:
        st.write("No Image Selected")