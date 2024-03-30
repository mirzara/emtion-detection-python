# Emotion Detection Using TensorFlow/Keras

## Required Dependencies:

1. OpenCV
   - OpenCV (Open Source Computer Vision Library) is a popular open-source computer vision and machine learning software library.
   - It provides various tools and functions for real-time image and video processing, including image capture, processing, and analysis.
   - Command: `pip install opencv-python`
   
2. NumPy
   - NumPy is a fundamental package for scientific computing with Python.
   - It provides support for multidimensional arrays and matrices, along with mathematical functions to operate on these arrays.
   - Command: `pip install numpy`
   
3. TensorFlow/Keras
   - TensorFlow is an open-source machine learning framework developed by Google.
   - Keras is an open-source neural network library written in Python that serves as a high-level API for TensorFlow.
   - In this project, TensorFlow and Keras are used to load a pre-trained model for real-time emotion detection.
   - Command: `pip install tensorflow`
   
4. scikit-learn
   - Scikit-learn is a popular machine learning library for Python.
   - It provides simple and efficient tools for data mining and data analysis, including classification, regression, clustering, and dimensionality reduction.
   - Command: `pip install scikit-learn`

**Setting Up a Virtual Environment (Optional):**

1. Virtualenv
   - Virtualenv is a tool used to create isolated Python environments.
   - It allows you to install dependencies for a project without affecting other projects or the system Python installation.
   - To create a virtual environment named "myenv", execute the following commands:
     ```
     pip install virtualenv
     virtualenv myenv
     ```
   - To activate the virtual environment on Windows:
     ```
     myenv\Scripts\activate
     ```
     On Unix/Linux:
     ```
     source myenv/bin/activate
     ```

Installation Instructions:

1. OpenCV, NumPy, TensorFlow, scikit-learn
   - Execute the respective `pip install` commands mentioned above to install each dependency.

Once you have installed the required dependencies, you can proceed to run the Real-Time Emotion Detection project using the provided Python script.

# Note on Google Colab
## Objective:
The objective of this project was to develop an emotion detection system using TensorFlow/Keras in Google Colab. The system aims to detect emotions such as anger, disgust, fear, happiness, sadness, surprise, and neutrality from images captured from a webcam.

## Key Steps:

### Data Collection and Preprocessing:
- We collected a dataset of facial images labeled with different emotions.
- Preprocessed the images by resizing them to a common size, converting to grayscale, and normalizing pixel values.
- We used CK+ dataset.

### Model Training:
- Built a convolutional neural network (CNN) model using TensorFlow/Keras to classify emotions from facial images.
- Trained the model on the collected dataset using Google Colab's GPU resources for faster training.

### Model Evaluation:
- Evaluated the trained model on a separate test dataset to assess its performance.
- Calculated metrics such as accuracy to measure the model's effectiveness in emotion detection.

### Real-Time Emotion Detection:
- Implemented real-time emotion detection using OpenCV to capture images from the webcam.
- Utilized the trained model to predict emotions from the captured images.
- Displayed the predicted emotions overlaid on the webcam feed for real-time feedback.

## Results:
The trained model achieved a test accuracy of approximately 91.88%, indicating its effectiveness in recognizing emotions from facial images. The real-time emotion detection system successfully detected and displayed emotions in real-time from webcam images.

## Conclusion: 
In conclusion, we developed an emotion detection system using TensorFlow/Keras in Google Colab, which can accurately classify emotions from facial images in real-time. This system has various potential applications in fields such as human-computer interaction, emotion analysis, and affective computing.

---

# Note on Real-Time Emotion Detection Python Script

## Objective:
The objective of this Python script is to perform real-time emotion detection using a pre-trained TensorFlow/Keras model and OpenCV to capture images from a webcam.

## Key Features:

### Model Loading:
- The script loads a pre-trained TensorFlow/Keras model for emotion detection. The model is assumed to be saved in the current directory.

### Image Preprocessing: 
- It preprocesses the captured webcam images by converting them to grayscale, resizing them to a common size, and normalizing pixel values.

### Emotion Detection:
- Utilizes the loaded model to predict emotions from the preprocessed images.
- The predicted emotions are overlaid on the webcam feed in real-time using OpenCV.

### Real-Time Webcam Feed:
- Captures images from the default webcam using OpenCV.
- Displays the webcam feed with predicted emotions overlaid for real-time feedback.

## Dependencies:
The script requires OpenCV and TensorFlow/Keras libraries to be installed.

## Usage:
To run the script, execute it in a Python environment with the necessary dependencies installed. Ensure that the pre-trained model file (trained_model.h5) is present in the same directory as the script.

## Conclusion:
This Python script provides a simple yet effective solution for real-time emotion detection using a webcam. It demonstrates the integration of machine learning models with computer vision techniques to enable real-time emotion analysis, which has various applications in areas such as human-computer interaction, sentiment analysis, and emotion recognition systems.

