import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def inpainting(image_path, prompt, x, y, width, height, output_path):
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y + height, x:x + width] = 255
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask)
    device = 'cpu'
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(device)
    edited_image = pipe(prompt=prompt, image=pil_image, mask_image=pil_mask).images[0]
    edited_image.save(output_path)

    return output_path