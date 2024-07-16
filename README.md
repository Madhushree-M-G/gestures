## Hand Gestures Using MediaPipe
This project demonstrates the recognition of hand gestures using MediaPipe, an open-source framework by Google for building perception pipelines.

## Introduction
This project aims to recognize various hand gestures using the MediaPipe framework. MediaPipe offers fast, accurate, and reliable hand tracking and gesture recognition using machine learning models.

![102222442-c452cd00-3f26-11eb-93ec-c387c98231be](https://github.com/user-attachments/assets/70b86c38-f126-4566-8442-bb0a02771dbc)


## Features
Real-time hand tracking and gesture recognition.
Support for various gestures (e.g., thumbs up, peace sign, etc.).
Easy to integrate into other projects.
Lightweight and efficient.

## Requirements
    mediapipe 0.8.1.
    OpenCV 3.4.2 or Later.
    Tensorflow 2.3.0 or Later.
    tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model).
    scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix).
    matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix).


## Directory
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│
└─utils
    └─cvfpscalc.py

##  app.py
This is a sample program for inference.
In addition, learning data (key points) for hand sign recognition,
You can also collect training data (index finger coordinate history) for finger gesture recognition.

## keypoint_classification.ipynb
This is a model training script for hand sign recognition.

## point_history_classification.ipynb
This is a model training script for finger gesture recognition.

## model/keypoint_classifier
This directory stores files related to hand sign recognition.
The following files are stored.

## Training data(keypoint.csv)
Trained model(keypoint_classifier.tflite)
Label data(keypoint_classifier_label.csv)
Inference module(keypoint_classifier.py)
model/point_history_classifier
This directory stores files related to finger gesture recognition.
The following files are stored.

## Training data(point_history.csv)
Trained model(point_history_classifier.tflite)
Label data(point_history_classifier_label.csv)
Inference module(point_history_classifier.py)

## utils/cvfpscalc.py
This is a module for FPS measurement.

## Reference
MediaPipe
Kazuhito00/mediapipe-python-sample
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)

## License
hand-gesture-recognition-using-mediapipe is under Apache v2 license.



