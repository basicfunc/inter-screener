import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tempfile
import ffmpeg

class AudioAnalyzer:
    def __init__(self, file):
        audio_path = self.preprocess_file(file)

        self.audio_path = audio_path
        self.audio, self.sr = librosa.load(audio_path)
        self.stft = librosa.stft(self.audio)
        self.mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr)
        self.pitches, self.magnitudes = librosa.piptrack(y=self.audio, sr=self.sr)
        self.magnitudes_db = librosa.amplitude_to_db(self.magnitudes)
        self.frames = range(self.pitches.shape[1])
        self.t = librosa.frames_to_time(self.frames, sr=self.sr)
        self.rms = librosa.feature.rms(y=self.audio, frame_length=len(self.audio) // 100)
        self.time_rms = librosa.frames_to_time(np.arange(len(self.rms[0])), sr=self.sr)
        self.confidence = np.mean(self.stft, axis=0) - np.std(self.stft, axis=0)
        self.time_confidence = librosa.frames_to_time(np.arange(len(self.confidence)), sr=self.sr)

    def get_rhythm_plot(self, ax):
        ax.plot(np.linspace(0, len(self.audio) / self.sr, len(self.audio)), self.audio)
        ax.set(title='Rhythm (Waveform)', xlabel='Time (s)', ylabel='Amplitude')

    def get_pause_silence_plot(self, ax):
        librosa.display.specshow(self.mfcc, x_axis='time', ax=ax)
        ax.set(title='Pause/Silence (MFCC)', xlabel='Time (s)', ylabel='MFCC Coefficients')

    def get_confidence_interval_plot(self, ax):
        ax.plot(self.time_confidence, self.confidence)
        ax.set(title='Confidence Interval', xlabel='Time (s)', ylabel='Confidence')

    def get_pitch_plot(self, ax):
        ax.plot(self.t, self.pitches.T, label='Pitch (Hz)')
        ax.set(title='Pitch', xlabel='Time (s)', ylabel='Pitch (Hz)')
        ax.set_xlim(0, len(self.audio) / self.sr)

    def get_spectrogram_plot(self, ax):
        librosa.display.specshow(self.magnitudes_db, sr=self.sr, x_axis='time', y_axis='log', ax=ax)
        ax.set(title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')

    def preprocess_file(self, video_path):
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            temp_wav_path = temp_file.name

        ffmpeg.input(video_path).output(temp_wav_path, acodec='pcm_s16le', ac=1, ar=16000).overwrite_output().run(quiet=True)

        return temp_wav_path

    def plot_all_charts(self):
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        self.get_rhythm_plot(axs[0, 0])
        self.get_pause_silence_plot(axs[0, 1])
        self.get_confidence_interval_plot(axs[0, 2])
        self.get_pitch_plot(axs[1, 0])
        self.get_spectrogram_plot(axs[1, 1])

        # Hide the empty subplot
        axs[1, 2].axis('off')

        # Adjust spacing between subplots
        fig.tight_layout()

        return fig

def calculate_silence_percentage(analyzer):
    total_duration = len(analyzer.audio) / analyzer.sr
    silence_duration = np.sum(analyzer.mfcc <= -50) / analyzer.sr
    percentage = (silence_duration / total_duration) * 100
    return percentage

def calculate_average_confidence(analyzer):
    magnitude = np.abs(analyzer.confidence)
    average_confidence = np.mean(magnitude)
    return average_confidence


def calculate_emphasis_percentage(analyzer):
    threshold = 1200  # Adjust the threshold as per your requirement
    emphasized_frames = np.sum(analyzer.pitches > threshold)
    total_frames = analyzer.pitches.shape[1]
    percentage = (emphasized_frames / total_frames) * 100
    return percentage


def calculate_speech_silence_ratio(analyzer):
    speech_duration = np.sum(analyzer.mfcc > -50) / analyzer.sr
    silence_duration = np.sum(analyzer.mfcc <= -50) / analyzer.sr
    ratio = speech_duration / silence_duration
    return ratio


def audioReport(analyzer):
    silence_percentage = calculate_silence_percentage(analyzer)
    average_confidence = calculate_average_confidence(analyzer)
    emphasis_percentage = calculate_emphasis_percentage(analyzer)
    speech_silence_ratio = calculate_speech_silence_ratio(analyzer)

    return silence_percentage, average_confidence, emphasis_percentage, speech_silence_ratio


if __name__ == '__main__':
    audio_path = 'input.mp4'

    analyzer = AudioAnalyzer(audio_path)