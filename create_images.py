from diffusers import StableDiffusionPipeline,EulerAncestralDiscreteScheduler
import glob
from pathlib import Path
from compel import Compel
import torch

def image_gen(height,width,n_steps):
    prompt_pathes=glob.glob('generated_prompt_for_SD/*')
    model_path=glob.glob('SD_model/*')[0]

    print(model_path)
    pipe = StableDiffusionPipeline.from_single_file(model_path).to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    if len(glob.glob('Lora/*'))>=1:
        pipe.load_lora_weights(glob.glob('Lora/*')[0])
        print(glob.glob('Lora/*')[0])

    negative_prompt='(worst quality,deformed, low quality,interlocked fingers,text,watermark)++++'
    negative_conditioning = compel.build_conditioning_tensor(text=negative_prompt)

    for prompt_path in prompt_pathes:
        with open(prompt_path,'r') as f:
            prompt= f.read()
            image=pipe(prompt, negative_prompt_embeds=negative_conditioning, num_inference_steps=n_steps, guidance_scale = 7.5, height=height,width=width).images[0]
            image.show()
            image.save(f'generated_thumbnails/{Path(prompt_path).stem}_thumbnail.png')
            f.close()

# print(image)
# img=Image.fromarray(image)
# image.show()

