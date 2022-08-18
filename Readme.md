
## SIGN LANGUAGE  ✊ ✌
                                

##  Index

- Description
- Demo
- How did I carry out the project?
- Implementation of project
- Dataset
- Objectives
- Acknowledgements
- Authors

## Demo

![video_readmegif](https://user-images.githubusercontent.com/101796802/185381629-111f051c-38e8-40b8-b280-ab70863219af.gif)

## Description

![picture_introduction](https://raw.githubusercontent.com/iremirezdeganuza72/sign_language/main/Imagenes/dibujos_manos.jpg)

Sign language is a project in which a Machine Learning model recognizes hand gestures and predicts the letters of the sign language alphabet in real time. It uses Python, Mediapipe, SKlearn, OpenCV and Streamlit.
## Demo

Insert gif or link to demo


## How did I carry out the project?

I used MediaPipe Hands that is a high-fidelity hand and finger tracking solution.
It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame.

![MediaPipe landmarks](https://raw.githubusercontent.com/iremirezdeganuza72/sign_language/main/Imagenes/hand_landmarks.png)

![MediaPipe real hands](https://raw.githubusercontent.com/iremirezdeganuza72/sign_language/main/Imagenes/hand_landmarks_2.png)

After downloading the dataset from Kaggle (https://www.kaggle.com/datasets/grassknoted/asl-alphabet) I extracted the 21 landmarks 
of each image. I also created my own dataset to have another alternative dataset in order to be able to work and train the ML models with both datasets.
After training different ML models; KNN, Random forest, SVC,... and after testing with the camera, the model that best predicts was Random forest.

It is important to note that there are several letters, namely A, L, T, Z, that none of the models read them well or predict them. 
In the case of the Random Forest model, there were also other letters such as U, V and the letters N and M that are very similar and it is difficult for them to predict correctly.

After saving the model, I created the file where I load the model and implement the camera so that streamlit can be restarted so that through streamlit a user can have the option to open the camera and be able to predict.

## Implementation of project

First of all, you have to clone the repository;

```bash
  git clone git@github.com:iremirezdeganuza72/sign_language.git
```

Secondly, it is necessary to install the different libraries includes in "requirements" file;

```bash
  pip install -r requirements.py
```

Afterwards, you can download the dataset from Kaggle, with 78000 images (https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
In case you want to create your own dataset, you should run "data_creation" file, executing the following command;

```bash
pip run data_creation.py
```

The next step would be to train and predict the different models and observing the metrics obtained, choosing the one that offers the best results against the prediction in front of the camera. 
In our case the best model is Random Forest, that has to be saved. It is saved as "RF_model.pkl".

Finally, to run streamlit, you must execute the following command

```bash
 streamlit run streamlit_camera.py
```

In case it does not open browser, it should appear a local address like this one;

![localhost](https://raw.githubusercontent.com/iremirezdeganuza72/sign_language/main/Imagenes/localhost.png)


## Dataset

I worked with a Kaggle dataset where are included 78000 images (https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
Apart from this dataset, I worked with my own dataset created through the frames obtained from a video, with which I achieved 100 images per letter, making a total of 2600 images in ttotal. 
## Objectives

The next steps in this project are firstly, to upload it to production through an api and with the Heroku cloud platform.
On the other hand, a second development of the project includes the written transcription of the different predicted letters, so that complete sentences can be realized.
## Acknowledgements

Core Code School

Alvaro Lucas

Daniel Alejandro Alvarado

Santino Lede

Marc Pomar

## Authors

This project was made by Iñigo Remirez de Ganuza;

- [@iremirezdeganuza72](https://github.com/iremirezdeganuza72/sign_language)

