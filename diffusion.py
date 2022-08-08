import torch 
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
from datasets import load_dataset
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
from networks import Unet
from schedules import *
from utils import num_to_groups
import tqdm.auto as tqdm
import torch 
from utils import *
from config import DEFAULT_CONFIG
import wandb 
from torchvision import datasets
from dataset import MyDataset
import os
from tqdm import tqdm

use_wandb = DEFAULT_CONFIG["WITH_WANDB"]
if not use_wandb:
    wandb.init(config=DEFAULT_CONFIG,project="diffusion_anime",name="test_0",mode="disabled")
else:
    wandb.init(config=DEFAULT_CONFIG,project="diffusion_anime",name="test_0")
    
timesteps = DEFAULT_CONFIG["TIMESTEPS"]
betas = linear_beta_schedule(timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas,axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1],(1,0),value=1.)
sqrt_recip_alphas = torch.sqrt(1./alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.-alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

reverse_transform = Compose([
        Lambda(lambda t: (t+1)/2),
        Lambda(lambda t: t.permute(1,2,0)),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

def q_sample(x_start,t,noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod,t,x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod,t,x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start,t):
    x_noisy = q_sample(x_start,t)
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image

def p_losses(denoise_model,x_start,t,noise=None,loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)
        
    x_noisy = q_sample(x_start,t,noise)
    predicted_noise = denoise_model(x_noisy,t)
    if loss_type == "l1":
        loss = F.l1_loss(noise,predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise,predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise,predicted_noise)
    else:
        return NotImplementedError
    return loss

def extract(a,t,x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1,t.cpu())
    return out.reshape(batch_size,*((1,)*(len(x_shape)-1))).to(t.device)

@torch.no_grad()
def p_sample(model,x,t,t_index):
    betas_t = extract(betas,t,x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod,t,x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas,t,x.shape)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x,t) / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance,t,x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
@torch.no_grad()
def p_sample_loop(model,shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape,device=device)
    imgs = []
    for i in tqdm(reversed(range(0,timesteps)),desc='sampling loop time step',total=timesteps):
        img = p_sample(model,img,torch.full((b,),i,device=device,dtype=torch.long),i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model,image_size,batch_size=16,channels=3):
    return p_sample_loop(model,shape=(batch_size,channels,image_size,image_size))

transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t*2)-1),
    ])

def transforms(examples):    
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

def train(epochs):
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
        for step,batch in progress_bar:
            optimizer.zero_grad()
            batch_size = batch.shape[0]
            batch = batch.to(device)
            t = torch.randint(0,timesteps,(batch_size,),device=device,dtype=torch.long)
            loss = p_losses(model,batch,t,loss_type="l1")
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            progress_bar.set_description(f"Loss: {loss.item()}")
            wandb.log({"loss":loss.item()})
            loss.backward()
            optimizer.step()
            
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4,batch_size)
                all_images_list = list(map(lambda n:sample(model,image_size,batch_size=n,channels=channels),batches))
                print(all_images_list)
                all_images = torch.cat(all_images_list,dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images,f"{results_folder}/sample-{milestone}.png",nrow=6)
                wandb.log({"sample": wandb.Image(all_images)})
                
if __name__ == "__main__":
    image_paths = os.listdir("../anime_face")
    dataset = MyDataset(image_paths=image_paths,transform=transform)
    dataset = torch.utils.data.Subset(dataset,np.linspace(0,len(dataset),1000,endpoint=False).astype(int))
    image_size = DEFAULT_CONFIG["IM_SIZE"]
    channels = DEFAULT_CONFIG["CHANNELS"]
    batch_size = DEFAULT_CONFIG["BATCH_SIZE"]
    save_and_sample_every = DEFAULT_CONFIG["SAVE_AND_SAMPLE_EVERY"]
    epochs = DEFAULT_CONFIG["EPOCHS"]
    
    #transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
    #dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    #batch = next(iter(dataloader))
    #print(batch.keys())

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1,2,4),
    ).to(device)
    optimizer = Adam(model.parameters(),lr=1e-3)
    
    train(epochs)
