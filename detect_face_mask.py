import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('face_mask_detection_model.h5')

# Function to preprocess the image for model input
def preprocess_image(image):
    image = cv2.resize(image, (136, 102))  # Resize to the model's input size
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (if necessary)
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # You would typically want to detect faces in the frame first.
    # Here is a simple Haar Cascade face detector.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        processed_face = preprocess_image(face)

        # Make predictions
        predictions = model.predict(processed_face)
        mask_class = np.argmax(predictions[0])  # Get the class with the highest probability

        # Define labels based on the predicted class
        label = 'Mask' if mask_class == 0 else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        # Draw rectangle around the face and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
