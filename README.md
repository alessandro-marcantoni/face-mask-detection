# Face-Mask-Detection

## Introduction
A *Face Mask Detector* built using only ```.tflite``` models. Designed to be lightweight and suitable for Edge Computing devices such as embedded systems and mobile devices.

## Webcam inference
In order to run the **live webcam inference** you need the following requirements:

* ```python3``` and ```pip3``` installed on your machine
* The libraries listed in ```requirements.txt```.  
  You can easily install them by typing from terminal
```bash
    pip3 install -r requirements.txt
```
Then you can simply run the program with
```bash
    python3 camera_infer_tflite.py
```

## Info about the inference method
The inference process is made up of two steps:

1. A first neural network detects the faces in every frame of the video stream.
2. A second neural network takes as input the faces detected in the previous step and figures out if they wear the face mask or not.

## Performances
The **webcam inference** has been tested on the following machines:

* Macbook Pro 2018 (i5@2.3GHz)
    * inference time: 0.020ms - 0.030ms
* iMac 27'' 2017 (i5@3.4GHz)
    * inference time: 0.015ms - 0.020ms

### Credits
* https://github.com/610265158/DSFD-tensorflow/tree/tf2 for the face detection model
* https://github.com/estebanuri/facemaskdetector for the face mask classification model
