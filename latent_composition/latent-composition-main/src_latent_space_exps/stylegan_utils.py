"""
Date: 29/12/21
This file has all the utility function to load the stylegan models, images and latents 
"""
import os
import torch
from torchvision import transforms 
from PIL import Image 
import sys

sys.path.append('../') 
from utils import show, renormalize, pbar  
from networks import networks 


os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/torch_cpp/' # needed for stylegan to run

# This function loads the image from a give path and resize and normalize it to outdim and create a pytorch tensor for processing
def load_image_tensor(img_path, outdim): 
    transform = transforms.Compose([
            transforms.Resize(outdim), 
            transforms.CenterCrop(outdim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    

    source_im = transform(Image.open(img_path))[None].cuda()
    return source_im

# Forward pass through the encode model to obtain latent codes. This is a direct encoding without the latent optimization, hence it is fast 
def encode_forward(nets, outdim, img_path): 
    img_save_size = 256
    source_im = load_image_tensor(img_path, outdim)

    # Image to be saved in small size 
    save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
    # Performing a forward pass of the encoder
    with torch.no_grad():
        # out = nets.invert(source_im)
        
        mask = torch.ones_like(source_im)[:, [0], :, :]
        z = nets.encode(source_im, mask)
        # print("z vector: ", z.shape)
        
    return z, save_src_img 


# This function will decode the latent code to syntehsize an image by having a forward pass through the StyleGAN2 model 
def decode_forward(nets, outdim, z): 
    with torch.no_grad():
        out_s = nets.decode(z)

    return out_s   


# This function loads the encoder and StyleGAN2 model and returns them 
def load_nets():  
    
    # bonus stylegan encoder trained on real images + identity loss
    # nets = networks.define_nets('stylegan', 'ffhq', ckpt_path='pretrained_models/sgan_encoders/ffhq_reals_RGBM/netE_epoch_best.pth')
    
    # stylegan trained on gsamples + identity loss
    nets = networks.define_nets('stylegan', 'ffhq')
    return nets 





