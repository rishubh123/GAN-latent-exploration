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
import pickle

# sys.path.append('../')   


# This function will create a forward pass and returns the image corresponding to the given latent code 
def infer_model(nets, outdim, latent):
    img = decode_forward(nets, outdim, latent)
    return img

# Given the set of latent paths this will read the latents and append them into a list to return all the group of latents. 
def read_latents(latent_paths):
    latents = []
    for i in range(len(latent_paths)):
        latent = np.load(latent_paths[i])
        latents.append(latent)
        # print("laoded latent shape:", latent.shape)

    return latents


# This is the main module which will create the synthesized image given the latent code from the styleflow module 
def run_main():
    # src_folder = '../../../../StyleFlow/StyleFlow/results/individual_latents'
    src_style_flow_imgs = '../../../../StyleFlow/StyleFlow/data/sg2latents.pickle'
    # load_latent_files = 
    loaded_src_latents = pickle.load(open(src_style_flow_imgs, "rb")) 
    print("latent: ", loaded_src_latents.keys())
    print("loaded latent shape: ", len(loaded_src_latents['Latent']))
    loaded_src_latents = loaded_src_latents['Latent']  

    dst_folder = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/styleflow_src_imgs/'   
    latent_folder = '../../data_files/comparison_files/InterfaceGAN/latent_directions/' 
    nets = load_nets() 
    outdim = 1024
    img_save_size = 1024
    n_attr = 5 

    alphas = [10.0, 3.0, -8.0, -0.5, -5.0]
    attr_list = ['angle_horizontal', 'gender', 'age', 'smile', 'glasses']
    latent_paths = [os.path.join(latent_folder, attr + '.npy') for attr in attr_list] 

    latent_dirs = read_latents(latent_paths) 

    # new_order = ['Pose', 'Expression', 'Gender', 'Age']
    for file_id in range(len(loaded_src_latents)):
        # fn = os.path.join(src_folder, file_) 
        latent_src = loaded_src_latents[file_id] # directyl reading the file from the list of loaded latent 
        output_imgs = [] 

        # Saving the source image as the first image into the stack
        orig_latent = torch.from_numpy(latent_src).cuda()
        input_img = infer_model(nets, outdim, orig_latent)
        input_img_norm = renormalize.as_image(input_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
        output_imgs.append(input_img_norm)

        for i in range(0, n_attr):
            latent_dir = latent_dirs[i] 
            w = latent_src + alphas[i]*latent_dir  
            latent = torch.from_numpy(w) 
            latent_torch = latent.type(torch.FloatTensor).cuda()
            img = infer_model(nets, outdim, latent_torch)   
            
            img_norm = renormalize.as_image(img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
            output_imgs.append(img_norm)


        output_stack = np.hstack(output_imgs)
        save_img = Image.fromarray(np.array(output_stack, np.uint8)).convert('RGB') 

        img_n = str(file_id) + '.jpg'
        img_fn = os.path.join(dst_folder, img_n)

        print("Saving the image at: ", img_fn)
        save_img.save(img_fn)


if __name__ == "__main__":
    run_main()