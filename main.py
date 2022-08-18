import os
import numpy as np
import cv2
import mediapipe as mp
from os import listdir
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
HEIGHT= 200
WIDTH= 200
# Indicate the path to access the directory
data_path="data"
data= os.listdir(data_path)
# Run each file through the path and save it in "sign_language"
# Run each image through the path then save it in IMAGE_FILES
IMAGE_FILES=[]
X=[]
y=[]
for folder in data:
  sign_language= os.listdir(f"{data_path}/{folder}")
  for images in sign_language:
      IMAGE_FILES.append(f"{data_path}/{folder}/{images}")
#Configuration to read images of hands;
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.image.png
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
      continue
    
    land= []
    for hand_landmarks in results.multi_hand_landmarks:
      for landmarks in hand_landmarks.landmark:
        land.append(landmarks.x)
        land.append(landmarks.y)
        land.append(landmarks.z)
         
    X.append(land)
    y.append(file.split("/")[1])
  X=np.array(X)
  y=np.array(y)
  np.save("data_np/x.npy", X) 
  np.save("data_np/y.npy", y)    

