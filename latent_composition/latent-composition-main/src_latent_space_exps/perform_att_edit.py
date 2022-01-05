"""
Date: 30/12/2021
This module implements the functionality to edit style of any given for any attribute by estimating the manifold for possible attribute styles.
We are looking at more exprimentations that can be used for estimating the attribute style manifold. 
"""

from ast import parse
import numpy as np 
import os
from PIL import Image 
import torch 
from stylegan_utils import encode_forward, decode_forward, load_nets   
from utils import renormalize
import pickle 
 

# computing average direction based on the interpolation weights for latents 
def compute_interpolate_dir(latents, weights):
    store = np.zeros(latents[0].shape, dtype = np.float16)
    for i in range(0, len(weights)):
        store += latents[i] * weights[i]              

    return store

# This function will parse the saved pairwise latent direction from a dictionary file and processed them. In the
# processing first all the pairwise latent directions for any attribute style is normalized to unit length and then
# averaged to form a set of averaged out latent directions for each style for the given attribute. 
def parse_latent_db(latent_db_path, n_pairs): 
    # Loading the pickle file and verifyieng if it matches with the saved file 
    print("Latent db reading from path: ", latent_db_path)
    load_file = open(latent_db_path,'rb') 
    latent_db = pickle.load(load_file)
    load_file.close()

    att_avg_latents = []
    # Iterating over all the attribute styles and their corresponding pair of original and transformed images to process the latent attributes 
    for k, all_dirs in latent_db.items():
        # Normalizing all the pairwise distances to have unit length before averaging
        # print("Key for att: ", k, " length of transforms: ", len(all_dirs))
        normed_latents = []
        for i in range(len(all_dirs)):
            lat_leng = np.linalg.norm(all_dirs[i]) 
            lat = all_dirs[i]/lat_leng
            # print("lat leng: ", lat_leng)
            normed_latents.append(lat)

        # Taking only the first n_imgs for taking the average to estimate the attribute directions 
        normed_latents = normed_latents[:n_pairs]
        normed_latents = np.array(normed_latents)
        avg_dir = np.mean(normed_latents, axis=0)

        # print("without averaging and after normlization shape: ", normed_latents.shape)
        # print("after averaging, shape of dir:", avg_dir.shape)
        att_avg_latents.append(avg_dir)

    print("number of att styles parsed: {}".format(len(att_avg_latents)))
    return att_avg_latents

# This function will create a basis vectors by taking a difference between all the latent codes corresponding to styles for any given attribute
def create_basis_latents(latent_db_processed): 
    # Taking the difference of the original latents from the 0th latent direction to obtain basis vectors
    basis_latents = [latent_db_processed[0] - latent_db_processed[i] for i in range(1, len(latent_db_processed))]
    return basis_latents 

# Editing the image given the image path, the name of attribute to be edited and the latent db file path and number of images used for averaging. 
def edit_image_interpolate_atts(nets, img_path, img_idx, img_transform_path, att, latent_db_path, n_pairs, n_transforms, alpha): 
    n_atts = 10
    fixed_scalar = 20
    outdim = 1024
    img_save_size = 256 
    latent_db_processed = parse_latent_db(latent_db_path, n_pairs) 

    n_atts = len(latent_db_processed)
    print("number of attribute styles in input: ", n_atts)

    # To obtain the basis latent vectors of the attribute style space, given the set of processed latent codes normalized followed by averaged. 
    basis_latents = create_basis_latents(latent_db_processed)

    # To compute the dc shift, we will average out all the directions for each attribute style 
    dc_shift = np.array(latent_db_processed).mean(axis=0) 
    print("dc shift vector shape: ", dc_shift.shape) 

    for id in range(0, n_transforms): 
        basis_coeffs = [round(np.random.uniform(-0.50, 0.50),2) for i in range(0,n_atts-1)]
        avg_latent_dir = compute_interpolate_dir(basis_latents, basis_coeffs)
        
        avg_latent_dir_shifted = avg_latent_dir + dc_shift
        latent_dir_tensor = torch.from_numpy(avg_latent_dir_shifted).cuda()

        z, save_src_img = encode_forward(nets, outdim, img_path)
        zT = z + alpha*fixed_scalar*latent_dir_tensor 

        out_z_img = decode_forward(nets, outdim, z)
        out_zT_img = decode_forward(nets, outdim, zT)
        
        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
        save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

        combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
        save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 

        fn = img_idx[:-4] + '_transformed_' + att + '_' + str(basis_coeffs) + '_w_' + str(alpha) + '.jpg'
        folder_name = os.path.join(img_transform_path, str(alpha))
        
        if (not os.path.exists(folder_name)):
            os.mkdir(folder_name)

        save_img_path = os.path.join(folder_name, fn) 

        print("saving image: ", save_img_path)
        save_img.save(save_img_path) 


# Editing images by performing interpolations between different attribute types to create a new attribute 
def edit_image_interpolate_atts_group(nets, img_path, img_idx, img_transform_path, att, latent_db_path, n_pairs, n_transforms, alphas): 
    n_atts = 10
    fixed_scalar = 20
    outdim = 1024
    img_save_size = 256 
    latent_db_processed = parse_latent_db(latent_db_path, n_pairs)           

    n_atts = len(latent_db_processed)
    print("number of attribute styles in input: ", n_atts)

    # Call to obatin the basis latent vectors which are created by taking difference between specific attribute styles 
    basis_latent = create_basis_latents(latent_db_processed)
    
    # Dc shift to map the transforms back to the space of the attribute 
    dc_shift = np.array(latent_db_processed).mean(axis=0) 
    print("dc shift vector shape: ", dc_shift.shape, dc_shift.min(), dc_shift.max())  

    image_matrix = []
    for id in range(0, n_transforms):
        # Source image decode latent code 
        z, save_src_img = encode_forward(nets, outdim, img_path)
        out_z_img = decode_forward(nets, outdim, z)
        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 

        # Iterating over image column 
        image_column = [save_src_img, save_out_z_img]

        # Iterating over all the values of alphas  
        for alpha in alphas:
            basis_coeffs = [round(np.random.uniform(-1.0, 1.0),2) for i in range(0,n_atts-1)] 
            avg_latent_dir = compute_interpolate_dir(basis_latent, basis_coeffs)

            avg_latent_dir_shifted = avg_latent_dir + dc_shift
            latent_dir_tensor = torch.from_numpy(avg_latent_dir_shifted).cuda()

            # Iterating over the set of all the interpolation values to be used for averaging 
            zT = z + alpha*fixed_scalar*latent_dir_tensor    

            out_zT_img = decode_forward(nets, outdim, zT)  
            save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
            image_column.append(save_out_zT_img)  

        image_column = np.vstack(image_column)
        image_matrix.append(image_column)  

    image_matrix = np.hstack(image_matrix)
    save_img = Image.fromarray(np.uint8(image_matrix)).convert('RGB') 

    fn = img_idx[:-4] + '_transformed_' + att + '_style_variations_' + str(alphas) + '.jpg'
    save_img_path = os.path.join(img_transform_path, fn) 

    print("saving image: ", save_img_path)
    save_img.save(save_img_path)  




# This function has the main functionality to read and image and create attribute style transformations over it. 
# Once we define the attributes to be considered and the paths for their latent codes then edit image can be called for
# each of the attribute separately. 
def edit_image_set_interpolate_atts():
    nets = load_nets()
    img_path_root = '../../CelebAMask-HQ/data_filtered/test500'
    img_transform_path_root = '../../CelebAMask-HQ/data_filtered/renew/results'
    latent_path_root = '../../data_files/estimated_att_styles'

    # Specifieng paths for the latent db for processesing 
    att_list = ['eye_g_style', 'hair_style']
    latent_db_list = ['latent_style_att_dir_db_eye_g_style.csv', 'latent_style_att_dir_db_hair_style.csv']
    latent_db_paths = [os.path.join(latent_path_root, ln) for ln in latent_db_list] 
    
    img_transform_paths = [os.path.join(img_transform_path_root, att) for att in att_list]

    img_idxs = [img for img in os.listdir(img_path_root)]
    # Randomly shuffle image_idxs:
    import random
    img_idxs = random.sample(img_idxs, len(img_idxs))

    img_paths = [os.path.join(img_path_root, img_id) for img_id in img_idxs]

    # Number of images to be processed, number of pairs to be used for computation and the required edit strength 
    n = 25
    n_pairs = 5
    n_transforms = 16
    alpha = 0.6
    alphas = [0.2, 0.3, 0.5, 0.8]
    print("Editing {} images".format(n))

    img_transform_path_root = '../../CelebAMask-HQ/data_filtered/renew/results'
    img_transform_paths = [os.path.join(img_transform_path_root, att + '_' + str(n_pairs)) for att in att_list]

    for i in range(0, n):  
        for att_j in range(1, 2): # Currently running just for the hair attribute 
            # edit_image_interpolate_atts(nets, img_paths[i], img_idxs[i], img_transform_paths[att_j], att_list[att_j], latent_db_paths[att_j], n_pairs, n_transforms, alpha)
            edit_image_interpolate_atts_group(nets, img_paths[i], img_idxs[i], img_transform_paths[att_j], att_list[att_j], latent_db_paths[att_j], n_pairs, n_transforms, alphas)

if __name__ == "__main__":   
  print("running main ...")
  print("editing image dirs ...")  

  print("editing image with interpolation of attributes") 
  edit_image_set_interpolate_atts()   