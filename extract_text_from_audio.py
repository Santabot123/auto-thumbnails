from faster_whisper import WhisperModel
import time
import torch
from pathlib import Path
import glob


def extractor(model_size = "large-v3"):

    videosPathes=glob.glob("videos/*")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if DEVICE == "cuda":
        model = WhisperModel(model_size, device=DEVICE, compute_type="int8_float16")
    else :
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    for videoPath in videosPathes:
        start_time = time.time()
        segments, info = model.transcribe(videoPath, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        alltext=''
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            alltext += segment.text+'\n'

        with open(f'text_from_videos/{Path(videoPath).stem}_text.txt', 'w') as f:
            f.write(alltext)
            f.close()
        with open(f'text_from_videos/{Path(videoPath).stem}_lang.txt', 'w') as f:
            f.write(info.language)
            f.close()
        print("--- %s seconds ---" % (time.time() - start_time))