import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("trained_model.h5")  # Assuming the model file is saved in the current directory

# Function to preprocess the input image
def preprocess_image(image):
    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    gray_image = cv2.resize(gray_image, (48, 48))
    gray_image = gray_image / 255.0
    gray_image = np.expand_dims(gray_image, axis=-1)
    gray_image = np.expand_dims(gray_image, axis=0)
    return gray_image

# Function to detect emotion
def detect_emotion(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_emotion_index = np.argmax(predictions)
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion_label = emotion_labels[predicted_emotion_index]
    return predicted_emotion_label

# Example of using OpenCV to capture images from webcam
try:
    cap = cv2.VideoCapture(0)  # Use the default webcam (change the argument for other cameras)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract the face region and detect emotion
            face_roi = frame[y:y+h, x:x+w]  # <-- Extract the face region from the original frame
            emotion = detect_emotion(face_roi)
            
            # Display the detected emotion label
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Display the frame with faces and emotions
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

