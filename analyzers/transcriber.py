import sys
import ffmpeg
import subprocess
import os

def clean():
    sys.stdout.write('\033[F')
    sys.stdout.write('\033[K')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise Exception("You only need to pass single argument.")

	input_file = sys.argv[1]

	if os.path.isfile('output.wav'):
		print("'output.wav' already exists. Deleting...")
		os.remove('output.wav')
		clean()

	ffmpeg.input(input_file).output('output.wav', ar=16000, ac=1, c='pcm_s16le', format='wav').run(quiet=True)
	input_file = 'output.wav'
	print(f"Converted to {input_file}")

	clean()
	whisper_path = 'bin/whisper.exe'
	whisper_model = 'models/whisper-small.bin'
	arguments = ['-m', whisper_model, '-f', input_file, '-otxt', '-of', 'transcribe']

	if os.path.isfile('transcribe.txt'):
		print("'transcribe.txt' already exists. Deleting...")
		os.remove('transcribe.txt')
		clean()

	print(f"Transcribing: {input_file} using whisper.")
	subprocess.run([whisper_path] + arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	clean()
	print(f"Written: 'transcribe.txt' -- done✅.")