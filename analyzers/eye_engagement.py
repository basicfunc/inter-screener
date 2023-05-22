import cv2
import dlib
from scipy.spatial import distance as dist
from tqdm import tqdm
import pandas as pd

class EyeEngagementAnalyzer:
    def __init__(self, video_path, predictor_path, engagement_thresholds):
        self.video_path = video_path
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.ear_sum = 0
        self.frame_count = 0
        self.pbar = None
        self.engagement_thresholds = engagement_thresholds
        self.engagement_data = []

    def eye_aspect_ratio(self, eye) -> float:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def process_video(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.pbar = tqdm(total=total_frames, desc='Processing Frames')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            current_frame_data = {
                'Timestamp': timestamp,
                'EyeEngagement': None,
                'EngagementLevel': None
            }

            for face in faces:
                landmarks = self.predictor(gray, face)

                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                self.ear_sum += avg_ear
                self.frame_count += 1

                current_frame_data['EyeEngagement'] = avg_ear * 100
                current_frame_data['EngagementLevel'] = self.calculate_engagement_level(avg_ear)

            self.engagement_data.append(current_frame_data)
            self.pbar.update(1)

        cap.release()
        self.pbar.close()

    def calculate_engagement_level(self, ear):
        if ear < self.engagement_thresholds['low']:
            return 'Low'
        elif ear < self.engagement_thresholds['medium']:
            return 'Medium'
        else:
            return 'High'

    def calculate_eye_engagement(self) -> float:
        overall_ear = (self.ear_sum / self.frame_count) * 100
        return overall_ear

    def save_data_to_dataframe(self, filename) -> None:
        df = pd.DataFrame(self.engagement_data)
        df.to_csv(filename, index=False)

    def data_to_dataframe(self, filename) -> pd.DataFrame:
        return pd.DataFrame(self.engagement_data)

if __name__ == "__main__":
    video_path = "input.mp4"
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"

    thresholds = {
        'low': 0.2,
        'medium': 0.4
    }

    analyzer = EyeEngagementAnalyzer(video_path, predictor_path, thresholds)
    analyzer.process_video()
    analyzer.calculate_eye_engagement()
    analyzer.save_data_to_dataframe("eye_engagement_data.csv")
