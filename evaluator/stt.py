import whisper
from gtts import gTTS
import os

from pydub import AudioSegment

import sounddevice as sd


def speech_to_text(file_name):
    model = whisper.load_model("base")

    audio = whisper.load_audio(file_name)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # options = whisper.DecodingOptions()
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    print("Spoken ", result.text)
    return result.text


def save_audio_to_mp3(filename, audio_data, sample_rate):
    # audio_data = (audio_data * 32767).astype(np.int16)  # Scale to 16-bit range
    audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    audio_segment.export(filename, format="mp3")


def record_audio(sample_rate, channels, duration):
    print("Started to record audio")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()
    return audio_data



def text_to_speech(text):
    output_file = 'output.mp3'
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    os.system(f'start {output_file}')
    os.system("afplay output.mp3")


def main():
    sample_rate = 44100  # 44.1 kHz
    channels = 1
    audio_data = record_audio(sample_rate, channels, 10)
    temp_filename = "test.mp3"
    save_audio_to_mp3(temp_filename, audio_data, sample_rate)
    answer = speech_to_text(temp_filename)
    # text = speech_to_text("test.mp3")
    # text_to_speech(text)

if __name__ == "__main__":
    main()



