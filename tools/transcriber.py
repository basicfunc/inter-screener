import openai
import tempfile
import ffmpeg
import whisper

class Transcriber:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None

    def preprocess_file(self, video_path):
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            temp_wav_path = temp_file.name

        ffmpeg.input(video_path).output(temp_wav_path, acodec='pcm_s16le', ac=1, ar=16000).overwrite_output().run(quiet=True)

        return temp_wav_path

    def transcribe_local(self, audio_file):
        if self.model is None:
            self.model = whisper.load_model("large")

        result = self.model.transcribe(audio_file)
        return result["text"]

    def transcribe(self, audio_file):
        openai.api_key = self.api_key

        with open(audio_file, "rb") as f:
            transcript = openai.Audio.translate("whisper-1", f)

        return transcript.get('text', ' ')

    def transcribe_audio_file(self, audio_file):
        if self.api_key is None or self.api_key.strip() == '':
            return self.transcribe_local(audio_file)
        else:
            return self.transcribe(audio_file)

class TranscriberWithAPI(Transcriber):
    def __init__(self, api_key):
        super().__init__(api_key=api_key)

class TranscriberWithoutAPI(Transcriber):
    def __init__(self):
        super().__init__()


def transcript(file, key=None):
	transcriber = None
	
	if key is not None and key.strip() != '':
	    transcriber = TranscriberWithAPI(api_key=key)

	else:
	    transcriber = TranscriberWithoutAPI()

	wav_file = transcriber.preprocess_file(file)
	
	transcription = transcriber.transcribe_audio_file(wav_file)

	return transcription
