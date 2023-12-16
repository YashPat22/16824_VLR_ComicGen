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
    canny_image.save(img2img_dir + "cannyimage_" + str(i)+".png")
    return canny_image

# model_id = ""
model_id_finetuned = "./type2_finetuned_model_prompt1"
# model_id = "runwayml/stable-diffusion-v1-5"

#Assigning the GPU to the variable device
device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')


# Generate images for testing style loss
##CHANGE THIS
save_comic_dir = "saveComic/ControlNetExp16/" #Input directory to save produced panels in
img2img_dir = save_comic_dir + "img2img/"

os.makedirs(save_comic_dir, exist_ok=True) 
os.makedirs(img2img_dir, exist_ok=True) 

gc.collect()
torch.cuda.empty_cache()



model_id_img2img = "./type2_finetuned_model_prompt1"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
).to(device)
img2img_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id_img2img,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(device)

# img2img pipe settings
img2img_pipe.scheduler = UniPCMultistepScheduler.from_config(img2img_pipe.scheduler.config)

prompts = ["Firefighter runs towards fire, high quality, good anatomy, detailed", "Firefighter saving child from fire, high quality, good anatomy, detailed"]


negativeprompts2 = "low resolution, bad anatomy, worst quality, low quality"

img2img_numImagesPerPrompt = 2
img2img_guidance_scale = 7.5
img2img_inf_steps = 100
# img2img_strength = 0.8

width = 512
height = 512
numPanels = 2

#Initialize some parameters
panel = 1
thresh = 100000 #Can play with this value as hyperparameter


img2img_list = []
toTensor = transforms.ToTensor()


for i in range(numPanels):

   file_name = "saveComic/txt2img/txt2img"+ "_"+str(i+1)+".png"
   txt2img_image = Image.open(file_name)
   txt2img_image = np.asarray(txt2img_image)
   
   print(txt2img_image.shape)
 
   # print(t)
   canny_image = canny(txt2img_image, i)
   #Now for generating img2img
   if panel == 1:
         batchedImgs = torch.zeros((img2img_numImagesPerPrompt, 3, width, height))
         contentLoss = ContentLoss(toTensor(txt2img_image),device)
         # txt2img
         while(True): 
            print("generating first panel")
            results = img2img_pipe(
               prompts[i] + ", nlwx style",
               canny_image,
               negative_prompt = negativeprompts2,
               num_inference_steps = img2img_inf_steps,
               guidance_scale  = img2img_guidance_scale,
               num_images_per_prompt = img2img_numImagesPerPrompt
                )
            if any(results.nsfw_content_detected):
                print("NSFW, regenerating")
                results = img2img_pipe(
                    prompts[i] + ", nlwx style",
                    canny_image,
                    negative_prompt = negativeprompts2,
                    num_inference_steps = img2img_inf_steps,
                    guidance_scale  = img2img_guidance_scale,
                    num_images_per_prompt = img2img_numImagesPerPrompt
                    )
            else:
               gc.collect()
               torch.cuda.empty_cache()
               break
            
         img2img_images = results.images
         for j in range(img2img_numImagesPerPrompt):
            img2img_images[j].save(save_comic_dir+f"candidate_{j+1}_for_panel{panel}.png")
            batchedImgs[j] = toTensor(img2img_images[j])
         #Now find content loss and find argmin
         loss = contentLoss.forward(batchedImgs)
         argmin = torch.argmin(loss)
         print("panel 1 generated")
         img2img_image = img2img_images[argmin]
         img2img_image = img2img_images[0]
         img2img_image.save(img2img_dir+"panel_1.png")
         img2img_list.append(img2img_image)
         panel += 1 
         del loss, img2img_images
   # Now for the rest of the images
   else:
      print(f"generating panel {panel}")
      while (True): #Checking for threshold
         batchedImgs = torch.zeros((img2img_numImagesPerPrompt, 3, width, height))
         perceptLoss = CustomPerceptualLoss(toTensor(img2img_image).unsqueeze(dim = 0), device)
         while(True):  #Checking for NSFW content
            results = img2img_pipe(
                    prompts[i] + ", nlwx style",
                    canny_image,
                    negative_prompt = negativeprompts2,
                    num_inference_steps = img2img_inf_steps,
                    guidance_scale  = img2img_guidance_scale,
                    num_images_per_prompt = img2img_numImagesPerPrompt
                    )
            if any(results.nsfw_content_detected):
               print("NSFW, regenerating")
               results = img2img_pipe(
                    prompts[i] + ", nlwx style",
                    canny_image,
                    negative_prompt = negativeprompts2,
                    num_inference_steps = img2img_inf_steps,
                    guidance_scale  = img2img_guidance_scale,
                    num_images_per_prompt = img2img_numImagesPerPrompt
                    )
            else:
               
               break
         img2img_images = results.images
         for j in range(img2img_numImagesPerPrompt):
            img2img_images[j].save(save_comic_dir+f"candidate_{j+1}_for_panel{panel}.png")
            batchedImgs[j] = toTensor(img2img_images[j])
         #calculate style loss and content loss 
         argmin, loss = perceptLoss.forward(batchedImgs, 15000, 0)
         print(f"Min Loss Index: {argmin}, loss: {loss}")
         if (loss[argmin]<thresh):
            print(f"panel {panel} generated")
            img2img_image = img2img_images[argmin]
            img2img_image.save(img2img_dir+f"panel_{panel}.png")
            img2img_list.append(img2img_image)
            panel += 1
            del loss, img2img_images
            break
         else:
            print("Did not satisfy loss threshold, regenerate")
   
      
# Combine images into panel

img2img_grid = make_image_grid(img2img_list, rows=1, cols=2)
img2img_grid.save(save_comic_dir + "img2img_grid.png")

