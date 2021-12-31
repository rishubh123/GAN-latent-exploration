"""
Date: 30/12/2021
This module implements the functionality to edit style of any given for any attribute by estimating the manifold for possible attribute styles.
We are looking at more exprimentations that can be used for estimating the attribute style manifold. 
"""

import numpy as np
import os
from PIL import Image 
import torch 
from stylegan_utils import encode_forward, decode_forward, load_nets   
from utils import renormalize
 

# computing average direction based on the interpolation weights for latents 
def compute_intepolate_dir(latents, weights):
    store = np.zeros(latents[0].shape, dtype = np.float16)
    for i in range(0, len(weights)):
        store += latents[i] * weights[i]              

    return store

# Editing images by performing interpolations between different attribute types to create a new attribute 
def edit_image_interpolate_atts(nets, img_path, img_idx, img_transform_path, latent_paths): 
    outdim = 1024
    img_save_size = 256 
    scale_factor = 18

    latents = []
    for lp in latent_paths:
        lat = np.load(lp)
        lat_leng = np.linalg.norm(lat) 
        lat = scale_factor * lat/lat_leng
        print("lat leng: ", lat_leng)
        latents.append(lat)


    basis_latents = [latents[0] - latents[1],
                     latents[0] - latents[2],
                     latents[1] - latents[3],
                     latents[6] - latents[1],
                     latents[0] - latents[7],
                     latents[6] - latents[2],
                     latents[3] - latents[8]]

    dc_shift = latents[0]

    """
    interpolation_values = [[1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.8, 0.2, 0.0, 0.0, 0.0],
                             [0.2, 0.8, 0.0, 0.0, 0.0],
                             [0.0, 0.5, 0.5, 0.0, 0.0],
                             [0.5, 0.0, 0.5, 0.0, 0.0],
                             [0.0, 0.5, 0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.5, 0.5, 0.0]] 
    """
    # exit()

    basis_coeffs = [[round(np.random.uniform(-0.35, 0.35),2) for i in range(0,7)] for j in range(0,10)]
    print("basis coeffs: ", basis_coeffs)
    
    # Iterating over the set of all the interpolation values to be used for averaging 
    for i in range(0, len(basis_coeffs)):
        weights = basis_coeffs[i]

        # Computing the average latent with the given set of weights 
        avg_latent_dir = compute_intepolate_dir(basis_latents, weights) 

        avg_latent_dir = avg_latent_dir + dc_shift
        # print("norm length of average latent: ", np.linalg.norm(avg_latent_dir))
        # exit()

        latent_dir_tensor = torch.from_numpy(avg_latent_dir).cuda() 
        # print("latent avg type: ", type(avg_latent_dir[0]), "single latent type: ", type(latents[0][0]))
        # exit()

        z, save_src_img = encode_forward(nets, outdim, img_path)

        alpha = 0.8
        zT = z + alpha*latent_dir_tensor  

        out_z_img = decode_forward(nets, outdim, z)
        out_zT_img = decode_forward(nets, outdim, zT)  

        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
        save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

        combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
        save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 

        fn = img_idx + '_transformed_interpolate_atts_average5_norm_basis_eg' + str(weights) + '_w_' + str(alpha) + '.jpg'
        folder_name = os.path.join(img_transform_path, str(alpha))
        
        if (not os.path.exists(folder_name)):
            os.mkdir(folder_name)

        save_img_path = os.path.join(folder_name, fn) 

        print("saving image: ", save_img_path)
        save_img.save(save_img_path) 




# Editing images by performing interpolations between different attribute types to create a new attribute 
def edit_image_interpolate_atts_group(nets, img_path, img_idx, img_transform_path, latent_paths): 
    outdim = 1024
    img_save_size = 256 
    scale_factor = 18           

    latents = []
    for lp in latent_paths:
        lat = np.load(lp)
        lat_leng = np.linalg.norm(lat)
        lat = scale_factor * lat/lat_leng
        # print("lat leng: ", lat_leng)
        latents.append(lat)


    basis_latents = [latents[0] - latents[1],
                     latents[0] - latents[2],
                     latents[1] - latents[3],
                     latents[6] - latents[1],
                     latents[0] - latents[5], # 7
                     latents[6] - latents[2],
                     latents[3] - latents[8]]

    dc_shift = latents[0]
    # exit()

    basis_coeffs = [[round(np.random.uniform(-0.35, 0.35),2) for i in range(0,7)] for j in range(0,10)]
    # print("basis coeffs: ", basis_coeffs)

    image_matrix = []
    # Source image decode latent code 
    z, save_src_img = encode_forward(nets, outdim, img_path)
    out_z_img = decode_forward(nets, outdim, z)
    save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 

    # Creating additional first row just to cover only single input source image
    first_row = np.ones((256, 10*256, 3)) * 255
    first_row[:256, :256, :] = save_out_z_img 
    image_matrix.append(first_row) 

    alphas = [0.8, 1.0, 1.2]    
    for alpha in alphas:
        # Iterating over the set of all the interpolation values to be used for averaging 
        image_row = []

        for i in range(0, len(basis_coeffs)):
            weights = basis_coeffs[i]

            # Computing the average latent with the given set of weights 
            avg_latent_dir = compute_intepolate_dir(basis_latents, weights) 
            avg_latent_dir = avg_latent_dir + dc_shift 

            latent_dir_tensor = torch.from_numpy(avg_latent_dir).cuda() 

            zT = z + alpha*latent_dir_tensor  

            out_zT_img = decode_forward(nets, outdim, zT)  
            save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
            image_row.append(save_out_zT_img)  

        image_row = np.hstack(image_row)
        image_matrix.append(image_row)  

    image_matrix = np.vstack(image_matrix)
    save_img = Image.fromarray(np.uint8(image_matrix)).convert('RGB') 

    fn = img_idx[:-4] + 'report_transform_imgs_test_att_interpolate_norm_basis' + '.jpg'
    save_img_path = os.path.join(img_transform_path, fn) 

    print("saving image: ", save_img_path)
    save_img.save(save_img_path)  


# Editing images with the .pkl latent directions estimated through various augmentations of the inputs 
def edit_image_set_interpolate_atts(): 
    nets = load_nets()
    img_path_root = '../CelebAMask-HQ/data_filtered/test_imgs'
    img_transform_path = '../CelebAMask-HQ/data_filtered/transform_imgs_test_interpolate_atts_average5_norm_basis_eg'  
    data_files_root = '../data_files' 

    img_idxs = [img for img in os.listdir(img_path_root)]
    img_paths = [os.path.join(img_path_root, img_id) for img_id in img_idxs]

    latent_dirs_pkl = 'all_latent_dirs_pairwise5_8167.pkl'
    # latent_dirs_pkl = 'all_latent_dirs_pairwise5_27995.pkl'
    latent_path = os.path.join(data_files_root, latent_dirs_pkl) 

    latent_paths = ['888','1565','2686','4081','4289','5220','6114','29778','29995']
    latent_paths = [lp + '__eye_g_transform_average_5_imgs_eyeglasses.npy' for lp in latent_paths]
    latent_paths = [os.path.join(data_files_root, lp) for lp in latent_paths]
    

    # Number of images to be processed 
    n = 10
    print("Editing {} images".format(n)) 

    # Images saving for multiple variations for a single image and creating a matrix collage out of it
    img_transform_path = img_transform_path = '../CelebAMask-HQ/data_filtered/report_transform_imgs_test_att_interpolate_norm_basis/'  

    for i in range(4, 4+n): 
        # edit_image_interpolate_atts(nets, img_paths[i], img_idxs[i], img_transform_path, latent_paths)  
        edit_image_interpolate_atts_group(nets, img_paths[i], img_idxs[i], img_transform_path, latent_paths)


if __name__ == "__main__":   
  print("running main ...")
  print("editing image dirs ...")  
  # edit_image_set()

  print("editing image with interpolation of attributes") 
  edit_image_set_interpolate_atts()   