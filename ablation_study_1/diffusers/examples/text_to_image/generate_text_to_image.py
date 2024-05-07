from diffusers import StableDiffusionPipeline
import torch
import os

epochs = 50
os.makedirs(f"sd-finetuned-model-{epochs}", exist_ok=True)
pipeline = StableDiffusionPipeline.from_pretrained(f"sd-finetuned-model-{epochs}", torch_dtype=torch.float16, use_safetensors=True).to("cuda")


for i in range(100):
    image = pipeline(prompt="Neutrophil").images[0]
    image.save(f"synthetic_images_{epochs}/image_{i}.png")
