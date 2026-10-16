#NLP of Nepal Audio files using mlx_audio and NLP models
#Source: https://github.com/Blaizzy/mlx-audio
#Change the audio file and directory

from mlx_audio.stt.generate import generate_transcription

result = generate_transcription(
    model="mlx-community/whisper-large-v3-turbo-asr-fp16",
    audio="/Users/shitalbhandary/Work/AU PhD HPE 2022/Interviews/1 Dr Paban Sharma/2_02-18_interview.mp3")

print(result.text)

from mlx_audio.stt import load

# Speech recognition
#model = load("mlx-community/Qwen3-ASR-1.7B-8bit")
#result = model.generate("/Users/shitalbhandary/Work/AU PhD HPE 2022/Interviews/1 Dr Paban Sharma/2_02-18_interview.mp3", language="English")
#print(result.text)

# Speech recognition model from Microsoft: gave time stamps too!
model = load("mlx-community/VibeVoice-ASR-4bit")
result = model.generate("/Users/shitalbhandary/Work/AU PhD HPE 2022/Interviews/1 Dr Paban Sharma/2_02-18_interview.mp3", language="English")
print(result.text)
