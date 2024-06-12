import time
import math
import ffmpeg

from faster_whisper import WhisperModel

import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

input_video = "eng.mp4"
input_video_name = input_video.replace(".mp4", "")

def extract_audio():
    extracted_audio = f"audio-{input_video_name}.wav"
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio


def transcribe(audio):
    model = WhisperModel("small", device='cpu', cpu_threads=4, compute_type='int8')
    segments, info = model.transcribe(audio)
    language = info[0]
    print("Transcription language", info[0])
    segments = list(segments)
    for segment in segments:
        # print(segment)
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))
    return language, segments


def run():
    output=''
    extracted_audio = extract_audio()
    print('start')
    language, segments = transcribe(audio=extracted_audio)

    with open('out.txt', 'w+') as o:
        for segment in segments:
            output += str(segment.text) + '.\n'
            print(f'{output}\n')
        o.write(output)

if __name__ == '__main__':
    run()