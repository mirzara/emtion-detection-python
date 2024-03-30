import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("trained_model.h5")  # Assuming the model file is saved in the current directory

def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Preprocess the grayscale image (e.g., resize, normalize pixel values)
    # You may need to adjust this based on how you trained your model
    # Example:
    gray_image = cv2.resize(gray_image, (48, 48))
    gray_image = gray_image / 255.0
    # Add a singleton dimension for the channel
    gray_image = np.expand_dims(gray_image, axis=-1)
    # Add batch dimension
    gray_image = np.expand_dims(gray_image, axis=0)
    return gray_image



# Function to detect emotion
def detect_emotion(image):
    # Preprocess the input image
    processed_image = preprocess_image(image)
    # Predict emotion using the loaded model
    predictions = model.predict(processed_image)
    # Get the index of the predicted emotion
    predicted_emotion_index = np.argmax(predictions)
    # Map the index to the corresponding emotion label (e.g., "Happy", "Sad", etc.)
    # You may need to define the mapping based on your model's output
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion_label = emotion_labels[predicted_emotion_index]
    return predicted_emotion_label

# Example of using OpenCV to capture images from webcam
cap = cv2.VideoCapture(0)  # Use the default webcam (change the argument for other cameras)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Perform emotion detection on the frame
    emotion = detect_emotion(frame)
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

