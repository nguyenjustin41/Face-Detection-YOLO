# YOLOv3_CV
An implementation of the Deep Neural Nets You Only Look Once (YOLOv3) Algorithm for object detection in Computer Vision using CMake to build CUDA.
The purpose of this project is to get an implementation version of how Object detection works in Python.
This code turns on your computer's webcam device and draws green squares over all detected objects that has over a 50% confidence rate.
It also tracks the frames per second on the top left of the program.

**CONTENT**
coco.names - lists all the labels used for implementation.
main.py - python file used to load YOLO algorithm with CUDA GPU, turns on webcam, and draws green squares around detected objects all while displaying current FPS.
yolov3.weights - weights used that should be installed from -> https://pjreddie.com/darknet/yolo/
yolov3.cfg - configuration file used

The video I used to install the build for the GPU was on -> https://www.youtube.com/watch?v=PlW9zAg4cx8&t=164s
GPU hardware: NVIDIA GeForce GTX 1660 SUPER 

Testing it on my computer gets me around 6-8 FPS

This project was for educational purposes and served as a great way to getting introduced into Computer Vision / Object Detection.

**CUDA versions:**
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:24:09_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0


**Dependencies:**

numpy==1.20.2
opencv-contrib-python==4.5.2.52
opencv-python==4.5.2.52
