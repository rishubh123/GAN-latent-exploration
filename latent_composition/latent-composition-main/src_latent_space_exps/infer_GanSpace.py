"""
Date: 18/01/2022
This module will just forward pass the latent codes extracted from the GanSpace model for StyleGAN2 on ffhq dataset and pass throught the pytorch stylegan generator
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

    print("latent for Gan Space dimentions ") 

    dst_folder = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/GanSpace/'    
    latent_path = '../../data_files/comparison_files/GanSpace/latent_directions/stylegan2-ffhq_style_ipca_c80_n1000000_w.npz'  
    latent_data = np.load(latent_path, allow_pickle=False) 
    print("latent data: ", latent_data)

    X_comp = latent_data['act_comp']
    X_mean = latent_data['act_mean']
    X_stdev = latent_data['act_stdev']
    Z_comp = latent_data['lat_comp'] 
    Z_mean = latent_data['lat_mean'] 
    Z_stdev = latent_data['lat_stdev']

    print("latent file for GanSpace shape: ", latent_data['lat_comp'].shape) 
    
    print("Z comp: ", Z_comp.shape)
    print("Z mean: ", Z_mean.shape)
    print("Z stdev: ", Z_stdev.shape) 
    n_layers = 18

    # Array where W+ latent directions will be save by replicating copies of W latent code 
    Z_reshaped = np.zeros((80, 18, 512))
    for i in range(0, 80):
        Z_list = [Z_comp[i, :, :] for k in range(0,18)]
        Z_stack = np.vstack(Z_list)

        # print("Z stack shape: ", Z_stack.shape)
        Z_reshaped[i, : , :] = Z_stack

    print("Z combined shape: ", Z_reshaped.shape)
    Z_filtered = [Z_reshaped[1,:,:], Z_reshaped[3,:,:], Z_reshaped[46,:,:], Z_reshaped[21,:,:], Z_reshaped[58,:,:], Z_reshaped[34,:,:]]  
                  # smile-46, bald-21, wrinkles-20, trimmed-beard-58, maskara beard-41, old_lady-34
    att_list = ['pose', 'eye_g', 'smile', 'bald', 'trimmed-beard', 'age']   

    nets = load_nets() 
    outdim = 1024
    img_save_size = 512
    n_attr = 6 

    alphas = [-2.0, 4.0, -2.0, 5.0, -4.0, 3.0] 
    attr_list = [str(k) for k in range(0, 80)]

    for file_id in range(25):
        # fn = os.path.join(src_folder, file_) 
        latent_src = loaded_src_latents[file_id] # directyl reading the file from the list of loaded latent 
        output_imgs = [] 

        # Saving the source image as the first image into the stack
        orig_latent = torch.from_numpy(latent_src).cuda()
        input_img = infer_model(nets, outdim, orig_latent)
        input_img_norm = renormalize.as_image(input_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
        output_imgs.append(input_img_norm)

        for i in range(0, n_attr):
            latent_dir = Z_filtered[i]
            w = latent_src + alphas[i]*latent_dir  
            latent = torch.from_numpy(w)
            latent_torch = latent.type(torch.FloatTensor).cuda()
            img = infer_model(nets, outdim, latent_torch)   
            
            img_norm = renormalize.as_image(img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)  
            output_imgs.append(img_norm)

        src_img = output_imgs[0]
        output_rem = output_imgs[1:]
        grid = []
        """
        # output_stack = np.hstack(output_imgs)
        print("output stack shape: ", len(output_rem)) 
        # Now we will create a matrix by splitting the stack into multiple image rows
        for k in range(0, 10):
            row = []
            for l in range(0, 8):
                current_img = output_rem[8*k + l]
                # print("current image shape: ", current_img.size)
                row.append(current_img)
            row_stack = np.hstack(row)
            # print("row shape: ", row_stack.shape)
            grid.append(row_stack)

        # Creating the matrix by taking vstack for all the rows 
        grid = np.vstack(grid)
        print("matrix shape: ", grid.shape)
        """

        grid = np.hstack(output_imgs)


        save_mat_img = Image.fromarray(np.array(grid, np.uint8)).convert('RGB') 
        save_orig_img = Image.fromarray(np.array(src_img, np.uint8)).convert('RGB')


        img_o = str(file_id) + '_orig_.jpg'
        img_s = str(file_id) + '_orig_transform.jpg'
        img_op = os.path.join(dst_folder, img_o)
        img_sp = os.path.join(dst_folder, img_s)


        # Saving both the image stack and the original image as separate image files 
        print("Saving the image at: ", img_op, img_sp)
        save_mat_img.save(img_sp)
        save_orig_img.save(img_op)


if __name__ == "__main__":
    run_main()