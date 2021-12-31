"""
Date: 29/12/2021 
This module has all the required functions to initiate a pre-trained StyleGAN2 encoder-decoder model and perform inference from these models given any input image
The primary utility of this file is to create a forward pass through the encode model for any given image extract the corresponding latent codes in W space of StyleGAN2,
these latent codes are further save as a numpy matrix for later use. 
We implement two methods to estimate the correct latent codes for any image:
- Direct forward pass from the StyleGAN2 encoder to optain code w (fast)
- Direct forward pass followed by optimization of the latent code w, to match the reconstructed and the original input image. The optimization is using LFBGS algorithmf (slow)
"""

import os
from PIL import Image
import torch 
import numpy as np 
from torchvision import transforms 
from stylegan_utils import load_image_tensor, encode_forward, decode_forward, load_nets
from utils import show, renormalize, pbar   



# Function to encode the given image into the latent space and save the w embeddings and the inverted image into desired folder 
def extract_and_save_embeddings(nets, outdim, img_path_set, dst_path_set, embds_path_set):  
    # use a real image as source and iterating over all the image from the img_path_set 
    for i in range(0, len(img_path_set)):
        img_path = img_path_set[i] # Image path 
        img_dst_path = dst_path_set[i] # Path to save inverted image 
        embds_dst_path = embds_path_set[i] # Path to save the latent embeddings 
        img_save_size = 128

        source_im = load_image_tensor(img_path, outdim) 
        save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS)  # Resizing the source image to lower size for saving 
                        
        # Performing a forward pass of the encoder and the decoder
        with torch.no_grad():             
            mask = torch.ones_like(source_im)[:, [0], :, :]   # Mask of all ones to be passed to compositional encoder 
            z = nets.encode(source_im, mask)
            out_s = nets.decode(z)  # Output of image inversion by two step encoding and decoding 

            print("z vector: ", z.shape)
            
            # Converting the latent into numpy vector 
            z_np = z.cpu().detach().numpy()
            print("saving latent: ", embds_dst_path)
            np.save(embds_dst_path, z_np) 

            save_out_s_img = renormalize.as_image(out_s[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

            # Combining the images, original and the inversion for visualization 
            combined_display_image = np.hstack([save_src_img, save_out_s_img]) 
            save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 
            
            print("saving image: ", img_dst_path)
            save_img.save(img_dst_path)  


# Extract embeddings for all the images in a folder, given a image set path, destination path for the inversion and the embeddings. 
def extract_latents(src_imgs_path, dst_inversion_path, dst_embds_path): 
    outdim = 1024
    nets = load_nets()

    print("Extracting latents from: ", src_imgs_path)  

    # Defining the file paths as a list to read and save the embeddings 
    img_names = [img for img in os.listdir(src_imgs_path) if img[-4:] == '.jpg']
    print("Saving latents for : {} images".format(len(img_names)))

    
    imgs_path_set = [os.path.join(src_imgs_path, img) for img in img_names]
    dst_path_set = [os.path.join(dst_inversion_path, img) for img in img_names]
    embds_path_set = [os.path.join(dst_embds_path, img[:-4]+'.npy') for img in img_names]  

    extract_and_save_embeddings(nets, outdim, imgs_path_set, dst_path_set, embds_path_set)  



if __name__ == "__main__":   
  print("running main ...")

  # 1.1 Extracting the latent directions for all the image in a given folder 
  print("extracting the latent directions ...")  
  root_path = '../../CelebAMask-HQ/data_filtered/renew'
  src_image_folder = os.path.join(root_path, 'augmentations/filtered_att_dirs_dataset/pose')
  dst_inversion_folder = os.path.join(root_path, 'inversions') 
  dst_embds_folder = os.path.join(root_path, 'latents') 

  extract_latents(src_image_folder, dst_inversion_folder, dst_embds_folder)    
  