from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, AutoPipelineForText2Image, UniPCMultistepScheduler
import torch
import cv2
from torchvision import transforms
from Perceptual_Loss import StyleLoss
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import os
from PIL import Image
import numpy as np
from Perceptual_Loss_V2 import ContentLoss, CustomPerceptualLoss
from diffusers.utils import make_image_grid
import gc


def canny(img, i):
    image = np.array(img)
    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save(txt2img_dir + "cannyimage" + str(i) + ".png")
    return canny_image


# model_id = ""
model_id_finetuned = "./type2_finetuned_model_prompt1"
# model_id = "runwayml/stable-diffusion-v1-5"

#Assigning the GPU to the variable device
device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')


# Generate images for testing style loss
save_comic_dir = "saveComic/" #Input directory to save produced panels in
txt2img_dir = save_comic_dir + "txt2img/"


os.makedirs(txt2img_dir, exist_ok=True) 

gc.collect()
torch.cuda.empty_cache()

text2img_pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
text2img_pipe.enable_model_cpu_offload()
text2img_pipe.enable_xformers_memory_efficient_attention()

prompts = ["Firefighter runs towards fire, realistic, high quality, good anatomy, detailed", "Firefighter saving child from fire, realistic, high quality, good anatomy, detailed"]


negativeprompts = "low resolution, bad anatomy, worst quality, low quality, low detail"


numPanels = 2
txt2img_inf_steps = 100
txt2img_numImagesPerPrompt = 1
txt2img_guidance_scale = 10
txt2img_inf_steps = 100
txt2img_list = []


for i in range(numPanels):
    results = text2img_pipe(prompt = prompts[i], negative_prompt = negativeprompts, height = 512, width = 512)
    txt2img_image = results.images[0]
    # canny_image = canny(txt2img_image)
    txt2img_image.save(txt2img_dir+ f"txt2img_{i+1}.png")
    txt2img_list.append(txt2img_image)


txt2img_grid = make_image_grid(txt2img_list, rows=1, cols=2)
txt2img_grid.save(save_comic_dir + "txt2img_grid.png")