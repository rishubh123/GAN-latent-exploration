import torch
import numpy as np
from utils import show, renormalize, pbar
from utils import util, paintwidget, labwidget, imutil
from networks import networks
from PIL import Image
import os
import skvideo.io
from torchvision import transforms
import time


def test_model(nets, outdim, img_path_set):  
    # use a real image as source
    for img_path in img_path_set:
        transform = transforms.Compose([
                        transforms.Resize(outdim),
                        transforms.CenterCrop(outdim),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])    
        source_im = transform(Image.open(img_path))[None].cuda()

        show(['Source Image', renormalize.as_image(source_im[0]).resize((256, 256), Image.LANCZOS)])
        
        # Performing a forward pass of the encoder and the decoder
        with torch.no_grad():
            out = nets.invert(source_im)
            
            mask = torch.ones_like(source_im)[:, [0], :, :]
            z = nets.encoder(source_im, mask)
            out_s = nets.decoder(z)  # Output of image inversion by two step encoding and decoding 

            show(['GAN Reconstruction direct', renormalize.as_image(out[0]).resize((256, 256), Image.LANCZOS)])
            show(['GAN Reconstruction stepwise', renormalize.as_image(out_s[0]).resize((256, 256), Image.LANCZOS)])


def load_nets():
    
    # bonus stylegan encoder trained on real images + identity loss
    # nets = networks.define_nets('stylegan', 'ffhq', ckpt_path='pretrained_models/sgan_encoders/ffhq_reals_RGBM/netE_epoch_best.pth')
    
    # stylegan trained on gsamples + identity loss
    nets = networks.define_nets('stylegan', 'ffhq')
    return nets


def main():
    outdim = 1024 # For faces 
    nets = load_nets()
    root_path = '../CelebAMask-HQ/'

    # Defining the source folder for reading the input images 
    imgs_path = os.path.join(root_path, 'data_filtered/img')
    imgs_path_set = [img for img in os.listdir(imgs_path)]
    imgs_path_set = imgs_path_set[:10] # Currently reading only 10 images for inference 

    test_model(nets, outdim, imgs_path_set)  