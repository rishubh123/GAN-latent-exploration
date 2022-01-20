"""
Date: 11/01/2022
This module will just forward pass the latent codes extracted from the styleflow optimization and pass throught the pytorch stylegan generator
for editing and synthesizing images, 
""" 


import os 
import numpy as np
import torch 
from stylegan_utils import encode_forward, decode_forward, load_image_tensor, load_nets 
from utils import renormalize, inversions 
from PIL import Image   

# sys.path.append('../')   


# This function will create a forward pass and returns the image corresponding to the given latent code 
def infer_model(nets, outdim, latent):
    img = decode_forward(nets, outdim, latent)
    return img


# This is the main module which will create the synthesized image given the latent code from the styleflow module 
def run_main():
    src_folder = '../../../../StyleFlow/StyleFlow/results/individual_latents'
    dst_folder = '../../../../StyleFlow/StyleFlow/results/individual_imgs/'
    nets = load_nets() 
    outdim = 1024
    img_save_size = 1024
    n_attr = 5

    # attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    # new_order = ['Expression', 'Pose', 'Age', 'Glasses']
    for file_ in os.listdir(src_folder):
        fn = os.path.join(src_folder, file_) 
        latents = np.load(fn)
        output_imgs = [] 

        for id in range(0, n_attr):
            latent_torch = torch.from_numpy(latents[id, :, :, :]).cuda()
            img = infer_model(nets, outdim, latent_torch)   
            
            img_norm = renormalize.as_image(img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
            output_imgs.append(img_norm)


        output_stack = np.hstack(output_imgs)
        save_img = Image.fromarray(np.array(output_stack, np.uint8)).convert('RGB') 

        img_n = file_[:-4] + '.jpg'
        img_fn = os.path.join(dst_folder, img_n)

        print("Saving the image at: ", img_fn)
        save_img.save(img_fn)


if __name__ == "__main__":
    run_main()