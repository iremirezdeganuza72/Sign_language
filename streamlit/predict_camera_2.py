import cv2 
import numpy as np
from sklearn.svm import SVC 
from predict_camera_1 import img_live
import pickle
# load model
with open('RF_model.pkl', 'rb') as f:
    rf = pickle.load(f)
cap = cv2.VideoCapture(0)   
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
    cv2.imshow('Signals', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
