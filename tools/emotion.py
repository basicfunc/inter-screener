import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class FERModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.expressions = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, (48, 48))
        normalized = resized_gray / 255.0
        reshaped = np.reshape(normalized, (48, 48, 1))
        return reshaped

    def predict_emotion(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=False)[0]
        expression_index = np.argmax(prediction)
        emotion = self.expressions[expression_index]
        return emotion


class EmotionAnalyzer:
    def __init__(self, model_path, video_path):
        self.model = FERModel(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.emotion_counts = {emotion: 0 for emotion in self.model.expressions.values()}
        self.emotions_over_time = []

    def analyze_video(self):
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            emotion = self.model.predict_emotion(frame)
            self.emotion_counts[emotion] += 1
            self.emotions_over_time.append(emotion)

            frame_count += 1

        self.cap.release()

    def calculate_emotion_percentages(self):
        emotion_percentages = {emotion: (count / self.total_frames) * 100 for emotion, count in self.emotion_counts.items()}
        return emotion_percentages

    def calculate_positive_negative_percentages(self):
        positive_emotions = ['Happy', 'Surprise']
        negative_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']

        positive_count = sum(self.emotion_counts[emotion] for emotion in positive_emotions)
        negative_count = sum(self.emotion_counts[emotion] for emotion in negative_emotions)

        positive_percentage = (positive_count / self.total_frames) * 100
        negative_percentage = (negative_count / self.total_frames) * 100

        return positive_percentage, negative_percentage

    def display_emotion_percentages(self):
        emotion_percentages = self.calculate_emotion_percentages()

        print("Emotion Percentages:")
        for emotion, percentage in emotion_percentages.items():
            print(f'{emotion:10}: {percentage:.2f}%')

    def display_positive_negative_percentages(self):
        positive_percentage, negative_percentage = self.calculate_positive_negative_percentages()

        print("\nTotal Emotion Percentages:")
        print(f'Positive Emotions: {positive_percentage:.2f}%')
        print(f'Negative Emotions: {negative_percentage:.2f}%')

    def plot_emotion_percentages(self, ax):
        emotion_percentages = self.calculate_emotion_percentages()

        emotions = list(emotion_percentages.keys())
        percentages = list(emotion_percentages.values())

        ax.bar(emotions, percentages)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Percentage')
        ax.set_title('Emotion Percentages')


    def plot_emotions_over_time(self, ax):
        colors = {'Angry': 'red', 'Disgust': 'purple', 'Fear': 'orange', 'Happy': 'green',
                  'Sad': 'blue', 'Surprise': 'pink', 'Neutral': 'gray'}

        x = range(1, self.total_frames + 1)
        y = self.emotions_over_time
        c = [colors[emotion] for emotion in y]

        ax.scatter(x, y, c=c)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Emotion')
        ax.set_title('Emotions Over Time')


    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        self.plot_emotion_percentages(axs[0])
        self.plot_emotions_over_time(axs[1])

        fig.tight_layout()

        return fig


if __name__ == '__main__':    
    # Example usage:
    model_path = 'models/fer_model.h5'
    video_path = 'input.mp4'

    analyzer = EmotionAnalyzer(model_path, video_path)
    analyzer.analyze_video()
    analyzer.display_emotion_percentages()
    analyzer.display_positive_negative_percentages()
    analyzer.plot_emotion_percentages()
    analyzer.plot_emotions_over_time()