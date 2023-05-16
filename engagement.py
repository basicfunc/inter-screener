import cv2
import numpy as np

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables for metrics
frame_count = 0  # Total frames processed
movement_count = 0  # Total movements detected
face_count = 0  # Total faces detected
eye_contact_count = 0  # Total frames with eye contact

def calculate_movement(frame1, frame2):
    # Convert frames to grayscale for simplicity
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the frames
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply a threshold to extract significant differences
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels (movement) in the thresholded image
    movement = cv2.countNonZero(threshold)

    return movement

def detect_faces(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

def detect_eyes(frame):
    # Convert frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return eyes

# Path to your video file
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret:
        break

    # Calculate movement between frames
    movement = calculate_movement(frame1, frame2)

    # Update metrics
    frame_count += 1
    movement_count += movement

    # Detect faces in the frame
    faces = detect_faces(frame1)

    # Update face count
    face_count += len(faces)

    # Detect eyes in the frame
    eyes = detect_eyes(frame1)

    # Check for eye contact by counting the presence of eyes
    if len(eyes) > 0:
        eye_contact_count += 1

    # Display frames (optional)
    cv2.imshow('Video', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate engagement metrics based on the collected data
total_frames = frame_count
average_movement = movement_count / total_frames
average_face_count = face_count / total_frames
eye_contact_percentage = (eye_contact_count / total_frames) * 100

print("Total Frames:", total_frames)
print("Average Movement:", average_movement)
print("Average Face Count:", average_face_count)
print("Eye Contact Percentage:", eye_contact_percentage)
