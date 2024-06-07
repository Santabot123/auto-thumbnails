from create_prompt import create_pipeline,create_prompt_for_SD
from create_images import image_gen
import torch
import gc
import os
import glob


height=512 # height of the image in pixels
width=int(8*round((16/9*height)/8)) # width of the image in pixels
n_steps=15 # number of steps for Stable Diffusion model
use_audio_content=False 
use_visual_content=True
clean=False # if True - will delete all files in "text_from_videos" and "generated_prompt_for_SD" directories

def clean_trash(folder_name):
    files = glob.glob(f'{folder_name}/*')
    for f in files:
        os.remove(f)


if use_audio_content:
    from extract_text_from_audio import extractor
    extractor('large-v3')
elif use_visual_content:
    from describe_frames import describe_frames
    describe_frames(interval=60)

if clean:
    clean_trash('text_from_videos')
    clean_trash('generated_prompt_for_SD')


gc.collect()
torch.cuda.empty_cache()
PIPELINE, GENERATION_ARGS =create_pipeline()
create_prompt_for_SD(PIPELINE, GENERATION_ARGS,lines_in_chunk=15)
gc.collect()
torch.cuda.empty_cache()
image_gen(height,width,n_steps)



