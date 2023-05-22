import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class AudioAnalyzer:
    def __init__(self, audio_path):
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

    def get_rhythm_plot(self):
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(self.audio) / self.sr, len(self.audio)), self.audio)
        ax.set(title='Rhythm (Waveform)', xlabel='Time (s)', ylabel='Amplitude')
        return fig

    def get_pause_silence_plot(self):
        fig, ax = plt.subplots()
        librosa.display.specshow(self.mfcc, x_axis='time', ax=ax)
        ax.set(title='Pause/Silence (MFCC)', xlabel='Time (s)', ylabel='MFCC Coefficients')
        return fig

    def get_confidence_interval_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_confidence, self.confidence)
        ax.set(title='Confidence Interval', xlabel='Time (s)', ylabel='Confidence')
        return fig

    def get_pitch_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.pitches.T, label='Pitch (Hz)')
        ax.set(title='Pitch', xlabel='Time (s)', ylabel='Pitch (Hz)')
        ax.set_xlim(0, len(self.audio) / self.sr)
        return fig

    def get_spectrogram_plot(self):
        fig, ax = plt.subplots()
        librosa.display.specshow(self.magnitudes_db, sr=self.sr, x_axis='time', y_axis='log', ax=ax)
        ax.set(title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')
        return fig


if __name__ == '__main__':
    audio_path = 'output.wav'

    analyzer = AudioAnalyzer(audio_path)

    # Get rhythm plot
    rhythm_plot = analyzer.get_rhythm_plot()

    # Get pause/silence plot
    pause_silence_plot = analyzer.get_pause_silence_plot()

    # Get confidence interval plot
    confidence_interval_plot = analyzer.get_confidence_interval_plot()

    # Get pitch plot
    pitch_plot = analyzer.get_pitch_plot()

    # Get spectrogram plot
    spectrogram_plot = analyzer.get_spectrogram_plot()

    # Display all the plots
    plt.tight_layout()
    plt.show()