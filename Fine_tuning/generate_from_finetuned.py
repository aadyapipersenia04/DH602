# from medmnist import BloodMNIST
# import numpy as np
# import matplotlib.pyplot as plt

from dataclasses import dataclass
# from torchvision import transforms
import torch
# from PIL import Image
# from diffusers import DDPMScheduler
from diffusers import UNet2DModel
# import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
# from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
# from pathlib import Path
import os
from accelerate import notebook_launcher
# import random

@dataclass
class TrainingConfig:
	output_dir = "/home/varad/test100/"
	# seed = 0
	eval_batch_size = 10
	num_batches = 10
	gpu_number = 2 # GPU to choose

      
def evaluate(config):
	pipeline = DDPMPipeline.from_pretrained('/home/varad/aadya/finetuning', use_safetensors=True).to(f"cuda:{config.gpu_number}")

	test_dir = os.path.join(config.output_dir, "samples")
	os.makedirs(test_dir, exist_ok=True)

	for b in range(config.num_batches):
		images = pipeline(
			batch_size=config.eval_batch_size,
			generator=torch.manual_seed(b), # Seed = b
		).images

		for i,img in enumerate(images):
			img.save(f"{test_dir}/{b*config.num_batches + i + 1}.png")
		
		print(f"Completed Batch {b+1} of {config.num_batches}")

	# # Make a grid out of the images
	# image_grid = make_image_grid(images, rows=4, cols=4)

	# # Save the images
	# image_grid.save(f"{test_dir}/1.png")

config = TrainingConfig()
args = (config, )

notebook_launcher(evaluate, args, num_processes=1)