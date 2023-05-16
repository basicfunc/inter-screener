import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Load the pre-trained FER model
model = tf.keras.models.load_model('models/fer_model.h5')

# Define the facial expression classes
expressions = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Load the video
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize variables to track the count of each emotion
emotion_counts = {expression: 0 for expression in expressions.values()}

# Define batch size for processing frames
batch_size = 64

# Process frames in batches
with tqdm(total=total_frames, unit='frames') as pbar:
    frame_buffer = []
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for the FER model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, (48, 48))
        normalized = resized_gray / 255.0
        reshaped = np.reshape(normalized, (48, 48, 1))
        frame_buffer.append(reshaped)

        # Process frames in batches
        if len(frame_buffer) == batch_size:
            batch = np.array(frame_buffer)
            predictions = model.predict(batch, verbose=0)
            predicted_expressions = np.argmax(predictions, axis=1)

            # Update the count of each emotion
            for expression_index in predicted_expressions:
                emotion_counts[expressions[expression_index]] += 1

            # Clear the frame buffer
            frame_buffer = []

        # Update the progress bar
        pbar.update(1)

# Process any remaining frames in the frame buffer
if len(frame_buffer) > 0:
    batch = np.array(frame_buffer)
    predictions = model.predict(batch, verbose=0)
    predicted_expressions = np.argmax(predictions, axis=1)

    # Update the count of each emotion
    for expression_index in predicted_expressions:
        emotion_counts[expressions[expression_index]] += 1

# Release the video capture
cap.release()

# Determine the predominant emotion
predominant_emotion = max(emotion_counts, key=emotion_counts.get)

# Print the result
for emotion, count in emotion_counts.items():
    p = (count / total_frames) * 100
    print(f'{emotion:10}: {p:.2f}%')

# Calculate positive and negative emotion percentages
positive_emotions = ['Happy', 'Surprise']
negative_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']

positive_count = sum(emotion_counts[emotion] for emotion in positive_emotions)
negative_count = sum(emotion_counts[emotion] for emotion in negative_emotions)

positive_percentage = (positive_count / total_frames) * 100
negative_percentage = (negative_count / total_frames) * 100

# Print the result
print("\nTotal:\n-----")
print(f'Positive Emotions: {positive_percentage:.2f}%')
print(f'Negative Emotions: {negative_percentage:.2f}%')