import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from math import floor
import glob
from pathlib import Path

video_paths = glob.glob("./videos/*.mp4")
def frame_generator(video_path, interval_seconds=60):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    frame_interval = int(fps * interval_seconds)
    print(f"Frame interval: {frame_interval}")

    frame_count = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        frame_count += frame_interval

    cap.release()
    print("Frames extraction completed.")

def get_number_of_frames(video_path, interval_seconds=60):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return floor(frame_count/frame_interval)

def describe_frames(interval=60):
    for video_path in video_paths:


        model_id = "vikhyatk/moondream2"
        revision = "2024-05-20"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision,torch_dtype=torch.float16,low_cpu_mem_usage=True).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        frame_descriptions=''

        for frame in tqdm(frame_generator(video_path, interval_seconds=interval), total=get_number_of_frames(video_path, interval_seconds=interval)):
            image = Image.fromarray(frame)
            enc_image = model.encode_image(image)

            frame_descriptions += model.answer_question(enc_image, "List all the objects in this image.", tokenizer)
            frame_descriptions += '\n'

        with open(f'text_from_videos/{Path(video_path).stem}_text.txt' , 'w') as f:
            f.write(frame_descriptions)
            f.close()







