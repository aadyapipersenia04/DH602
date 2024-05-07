import torch.nn as nn
from medmnist import BloodMNIST
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from dataclasses import dataclass
import torch
from diffusers import UNet2DModel
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from diffusers import DDPMPipeline

# from diffusers.utils import make_image_grid
from typing import List
import PIL

from diffusers.utils import *
from accelerate import notebook_launcher

def make_image_grid(images: List[PIL.Image.Image], rows: int, cols: int, resize: int = None) -> PIL.Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        
    return grid
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([1],dtype=torch.float32).to(device)
device
class MyResnext50(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyResnext50, self).__init__()
        self.pretrained = my_pretrained_model
        self.pretrained.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(2048, 100),
                                nn.ReLU(),
                                nn.Linear(100, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.pretrained(x) 
        output = self.fc(x)
        return output
# Load the trained model
resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
reward_model = MyResnext50(my_pretrained_model=resnext50_pretrained)
reward_model.load_state_dict(torch.load('/home/varad/aadya/results/saved_models/best_model_ablation_3.pth',map_location=device))
reward_model.eval()  # Set the model to evaluation mode
# Forward pass to get the predicted labels
@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 30
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 2
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "/home/varad/aadya/finetuning_ablation_study"  # the model name locally and on the HF Hub
    temp_dir = "temp"  # where to save the model temporarily before using it with the reward model
    init_dir = "/home/varad/aadya/finetuning_ablation_study" # where the pretrained model is saved

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "varaddesh/ddpm-granulocytes"  # the name of the repository to create on the HF Hub
    # token = access_token
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
super_dataset = BloodMNIST(size = 64, split="train", download=True, transform=preprocess)
print(super_dataset)

indices = [i for i, (data, target) in enumerate(super_dataset) if target == 6] #neutrophils


train_dataset = torch.utils.data.Subset(super_dataset, indices)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, neutrophil_dataset, transforms=None):
        super(CombinedDataset, self).__init__()
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)
        self.neutrophil_dataset = neutrophil_dataset
        self.transforms = transforms 
        
    def __len__(self):
        return len(self.image_list) + len(self.neutrophil_dataset)
    
    def __getitem__(self, idx):
        
        if idx < len(self.neutrophil_dataset):
            return (self.neutrophil_dataset[idx][0], 0) # Training dataset
        idx = idx - len(self.neutrophil_dataset)
        image_path = os.path.join(self.folder_path, self.image_list[idx])
        image = Image.open(image_path)
        image_tensor = self.transforms(image)
        return (image_tensor, 1) # Synthetic dataset
combined_dataset = CombinedDataset('/home/varad/Generated-Images/samples', train_dataset, transforms=preprocess)
combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=config.train_batch_size, shuffle=True)
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    # num_training_steps=(len(train_dataloader) * config.num_epochs),
    num_training_steps=(len(combined_dataloader) * config.num_epochs),
)
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
def train_loop(config, model, noise_scheduler, optimizer, combined_dataloader, lr_scheduler, beta, weights, reward_model, power_factor):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    # model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, lr_scheduler
    # )
    
    model, optimizer, combined_dataloader, lr_scheduler, weights, reward_model = accelerator.prepare(
        model, optimizer, combined_dataloader, lr_scheduler, weights, reward_model
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar = tqdm(total=len(combined_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(combined_dataloader):
            clean_images = batch[0]
            syn_or_real = batch[1]
            # print(syn_or_real)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # pipeline2 = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                # # if(epoch == 0):
                # #     pipeline2 = DDPMPipeline.from_pretrained(config.init_dir, use_safetensors=True).to(device)
                # # else:
                # #     pipeline2 = DDPMPipeline.from_pretrained(config.temp_dir, use_safetensors=True).to(device)
                # gen_imgs = np.array(pipeline2(batch_size=bs,generator=torch.manual_seed(config.seed),num_inference_steps=1,output_type="numpy").images)
                # gen_imgs = torch.tensor(gen_imgs, dtype=torch.float32).to(device)
                # gen_imgs = torch.permute(gen_imgs, (0,3,1,2))
                
                
               

            with accelerator.accumulate(model):
                # Part added for finetuning------------------------------------------------

                
                # gen_imgs = np.array(pipeline(batch_size=bs,generator=torch.manual_seed(config.seed),output_type="numpy").images)

                # print(type(gen_imgs))

                

                # print("\n\n\nShape = ", gen_imgs.size(), "\n\n\n")

                rewards = reward_model(clean_images)                          # 1 when plausible
                # print("\n\n\nShape = ", rewards.size(), "\n\n\n")
                # print(rewards) 
                penalties = 1-rewards                                       # 0 when plausible
                # penalties = rewards                                           # 1 when plausible
                print(penalties.device, weights.device)
                weighted_penalties = torch.matmul(penalties, weights)
                weighted_penalties = weighted_penalties.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                weighted_penalties = torch.exp(power_factor*weighted_penalties) - 1 # Exponential scaling
                # print("\n\n\nShape = ", weighted_penalties.size(), "\n\n\n")
                weighted_penalties = weighted_penalties.to(device)
                # -------------------------------------------------------------------------

            # with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = 0.0
    
                for i in range(len(syn_or_real)):
                    if syn_or_real[i] == 1:                                             #synthetic
                        loss += F.mse_loss(noise_pred[i]*weighted_penalties[i], noise[i]*weighted_penalties[i])
                    else:                                                               #real
                        loss += beta * F.mse_loss(noise_pred[i], noise[i])
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
            # pipeline.save_pretrained(config.temp_dir)
beta = 1
power_factor = 4
args = (config, model, noise_scheduler, optimizer, combined_dataloader, lr_scheduler, beta, weights, reward_model, power_factor)

notebook_launcher(train_loop, args, num_processes=1) #num_processes is the number of GPUs used
