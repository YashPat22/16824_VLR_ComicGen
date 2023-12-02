from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms
from loss import StyleLoss
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import os
from combinePanels import combinePanels

model_id = "./path-to-save-model"
#model_id = "CompVis/stable-diffusion-v1-4"

#Assigning the GPU to the variable device
device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')


# Generate images for testing style loss

# prompt = ["city in cimoc style","realistic city", "city in cimoc style"]
# pipe = StableDiffusionPipeline.from_pretrained(model_id, 
#                                             #    safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
#                                                torch_dtype=torch.float16).to(device)

# results = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images

# results[0].save("template.png")
# results[1].save("realism.png")
# results[2].save("style.png")

# batchedImgs = torch.zeros((2, 3, 512, 512))
# toTensor = transforms.ToTensor()
# Style = StyleLoss(toTensor(results[0]).unsqueeze(dim = 0), device = device)
# batchedImgs[0] = toTensor(results[1])
# batchedImgs[1] = toTensor(results[2])


# min, loss = Style.find_style(batchedImgs)

# print(min)

# print(loss)

save_comic_dir = "saveComic/" #Input directory to save produced panels in
os.makedirs(save_comic_dir, exist_ok=True) 

pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                            #    safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                                               torch_dtype=torch.float16).to(device)
prompt = ["whitedog running in cimoc style", "white dog catching ball in cimoc style", "white dog brings ball back to owner in cimoc style","white dog happy in cimoc style"]

numImagesPerPrompt = 4
numPanels = 4
width = 512
height = 512

#Initialize some parameters
panel = 1
thresh = 105 #Can play with this value as hyperparameter

toTensor = transforms.ToTensor()
for i in range(numPanels):
   if panel == 1:
      while(True): #Check for any NSFW images
         results = pipe(prompt[panel-1], num_inference_steps=50, guidance_scale=10, height = height, width = width)
         if any(results.nsfw_content_detected):
            print("NSFW, regenerating")
            results = pipe(prompt[panel-1], num_inference_steps=50, guidance_scale=10, height = height, width = width) #Generate first image
         else:
            break
      print(f"panel 1 generated")
      image = results.images[0]
      Style = StyleLoss(toTensor(image).unsqueeze(dim = 0), device = device) #Calculate target features
      image.save(save_comic_dir+"panel_1.png")
      panel += 1 
      continue

   # Now for the rest of the images
   while (True):
      batchedImgs = torch.zeros((numImagesPerPrompt, 3, width, height)) #512 by 512 is default size, we can decrease size if we want
      
      while(True): ## Keep generating until there are no NSFW images
         results = pipe(prompt[panel-1], num_inference_steps=50, guidance_scale=10, num_images_per_prompt = numImagesPerPrompt, height = height, width = width)
         if any(results.nsfw_content_detected):
            print("NSFW, regenerating") 
            results = pipe(prompt[panel-1], num_inference_steps=50, guidance_scale=10, num_images_per_prompt = numImagesPerPrompt, height = height, width = width)   
         else:
            print(f"panel {panel} generated")
            break
      images = results.images
      for i in range(numImagesPerPrompt):
         images[i].save(f"candidate_{i+1}_for_panel{panel}.png")
         toTensor = transforms.ToTensor()
         batchedImgs[i] = toTensor(images[i])

      min_idx, loss = Style.find_style(batchedImgs)
      print(f"Min Loss Index: {min_idx}, loss: {loss}")
      if (loss[min_idx]<thresh):
         images[min_idx].save(save_comic_dir+f"panel_{panel}.png")
         panel += 1
         min_loss = 10
         break
      else:
         print("Did not satisfy loss threshold, regenerate")
      
# Combine images into panel
combinePanels("saveComic", prompts=prompt)
