# Extracting Face with Mediapipe


## Description
This project uses Mediapipe's face mesh model and selfie segmentation to extract the person's face present in the image. It detects the face landmarks, creates a mask around the face, and applies the segmentation mask to replace the background. The project is implemented in Python using OpenCV, Mediapipe, and NumPy libraries.

## Features
1. Detects face landmarks using Mediapipe's face mesh model.
2. Creates a binary mask around the face using the convex hull of the landmarks.
3. Applies selfie segmentation to segment the person's face from the image.
4. Applies AND operation on both mask
5. Extract the image from original image using mask.
6. Provides visualizations of the mask, face extraction, and final output.

## Installation
1. Clone the repository: git clone https://github.com/RaoSharjeelKhan/Extracting-Face-With-Mediapipe.git
2. Install the required dependencies: pip install -r requirements.txt
