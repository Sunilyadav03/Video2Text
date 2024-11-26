from moviepy.editor import VideoFileClip
import speech_recognition as sr
import re
import os
import whisper

def extract_audio_from_video(video_path):           #This extract_audio_from_video() converts our inputed video file to its corresponding audio file.
    # Load video and extract audio as .wav file
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path


def transcribe_audio(audio_path):                  #this transcribe_audio() function return transcript from the Audio file.
    # Check if the audio file exists
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"The audio file '{audio_path}' does not exist.")

    # Load the Whisper model
    model = whisper.load_model("medium")           # Choose the model size based on your needs

    # Transcribe the audio
    result = model.transcribe(audio_path)

    # Return the transcript
    transcript = result['text']
    return transcript

# Example usage
audio_path = "path_to_your_audio_file.wav"       # Change this to your audio file path
try:
    transcript = transcribe_audio(audio_path)
    print(transcript)
except Exception as e:
    print(f"Error: {e}")




def main(video_path):
    # Step 1: Extract and transcribe audio
    audio_path = extract_audio_from_video(video_path)
    transcript = transcribe_audio(audio_path)

    return transcript

# Example usage
video_path = 'path of your video file'
result = main(video_path)                        # "result" is the ultimate transcripte of our inputed video.
print(result)       
