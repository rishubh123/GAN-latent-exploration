"""
Date: 29/12/2021
This module has the functionality of creating the desired attribute edits for any given input image.
We leverage the latent direction estimated previously and use this direction to edit any new input image.
Specifically, input image is first encoded using the StyleGAN2 encoder into a latent code w. This latent code
is then transformed to obtain a new latent code w' which is then finally used to synthesize an image using StyleGAN2 generator model.
""" 

import os 
import numpy as np
import torch 
from stylegan_utils import encode_forward, decode_forward, load_image_tensor, load_nets 
from utils import renormalize, inversions 
from PIL import Image 
# sys.path.append('../')   

# This function performs normalized the latent code differences to unit length 
def normalize_dirs(latent_db):  
    latent_db_norm = []

    # Iterating over all the rows where the latent differences are stored
    for idx in range(0, latent_db.shape[0]):        
        # print("latent db shape: ", latent_db.shape)
        latent_dir = latent_db[idx,:,:,:] 

        magnitude = np.linalg.norm(latent_dir)
        # print("Magnitute for the latent direction: ", magnitude)
        latent_norm = latent_dir / magnitude
        latent_db_norm.append(latent_norm)

    return np.array(latent_db_norm)


# This function performs image editing, given any image path, the attribute latent db, number of pairs to use and the edit stregnth alpha 
def edit_image(nets, img_src_root, img_res_path, img_name, att, latent_db_path, alpha, dir, n_pairs, z_optimize):
    outdim = 1024
    img_save_size = 256 # Image resolution at which the inversion and transformed image stack will be dumbed for visualization 
    fixed_scalar = 10 # Fixed scalar to map the unit normalized vectors back to a meaningful value 
    img_path = os.path.join(img_src_root, img_name)

    print("loading att latent dir file: ", latent_db_path)
    latent_dir_db = np.load(latent_db_path) 
    latent_dir_db_norm = normalize_dirs(latent_dir_db)               

    # Filtering out the first set of 5 latents for processing
    latent_dir_db_norm = latent_dir_db_norm[:n_pairs, ...]

    # Average latent direction to be used for image editing 
    avg_latent_dir = latent_dir_db_norm.mean(axis=0)
    avg_latent_dir = torch.from_numpy(avg_latent_dir).cuda()  

    # If this flag is present then the latent code of the input image will be optimized first then the latent code transformation will be performed. 
    if (z_optimize):
        source_im = load_image_tensor(img_path, outdim) 
        att_stength = alpha*fixed_scalar

        # Note: There is some code issue for inversion optimization, have to recheck and rectify
        # Performing the latent optimization to estimate the identity preserved direction 
        # checkpoint_dict, opt_losses = inversions.optimize_latent_for_id(nets, source_im, avg_latent_dir, att_stength) 
        checkpoint_dict, opt_losses = inversions.invert_lbfgs(nets, source_im, mask=None, lambda_f=0.25, lambda_l=0.5, num_steps=30, initial_latent=None)
        
        # Image to be saved in small size 
        save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
        out_z_img = checkpoint_dict['current_x'].detach().clone() 
        z = checkpoint_dict['current_z'].detach().clone()

        zT = z + alpha*fixed_scalar*avg_latent_dir
        out_zT_img = decode_forward(nets, outdim, zT)

    else:
        z, save_src_img = encode_forward(nets, outdim, img_path)

        # Changing the z vector for edit based on the edit strength and the fixed scalar values 
        
        # Transfromation of z for changing the latent in positive direction if dir is positive  
        if (dir == 'pos'):
            zT = z + alpha*fixed_scalar*avg_latent_dir 

        # IF the direction is negative then we will subtract the direction from the current latent code for transformation 
        if (dir == 'neg'):
            zT = z - alpha*fixed_scalar*avg_latent_dir
        
        out_z_img = decode_forward(nets, outdim, z)
        out_zT_img = decode_forward(nets, outdim, zT)
    
    # Mapping the outpus generated to an image
    out_img_hr = renormalize.as_image(out_zT_img[0]).resize((outdim, outdim), Image.LANCZOS) 
    save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
    save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
    
    # Creating a combined stack image that will be saved for analysis and visualization 
    combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
    save_img = Image.fromarray(np.array(combined_display_image, np.uint8)).convert('RGB')

    fn_combined = img_name[:-4] + '_stack_' + att + '_' + dir + '_alpha_' + str(alpha) + '_' + str(z_optimize) + '.jpg'  # removing .jpg
    fn_res = img_name[:-4] + '_transformed_' + att + '_' + dir + '_alpha_' + str(alpha) + '_'+ str(z_optimize) + '.jpg'  

    # Saving the transformed images and the stack of transformed and original image 
    fn_combined_save_path = os.path.join(img_res_path, fn_combined)
    fn_res_save_path = os.path.join(img_res_path, fn_res) 

    print("saving image stack at:", fn_combined_save_path)
    save_img.save(fn_combined_save_path)
    # out_img_hr.save(fn_res_save_path) 
  

# Edit image function for generating results for a group of images into a single matrix form with varyieng the value of strength parameter alpha. 
# In this method we have not implemented the functionality of latent optimization for better inversion. 
def edit_image_group(nets, img_src_root, img_res_path, img_names, att, latent_db_path, alpha, z_optimize):
    outdim = 1024
    img_save_size = 256 

    print("loading att latent dir file: ", latent_db_path)        
    latent_dir_db = np.load(latent_db_path)
    latent_dir_db_norm = normalize_dirs(latent_dir_db) 

    # Average latent direction to be used for image editing 
    avg_latent_dir = latent_dir_db_norm.mean(axis=0)
    avg_latent_dir = torch.from_numpy(avg_latent_dir).cuda() 

    # number of images to be edited 
    n = len(img_names)
    image_matrix = []
    for i in range(n):  
        # Image path that is created by joining the src folder and the corresponding image name 
        img_path = os.path.join(img_src_root, img_names[i])
        z, save_src_img = encode_forward(nets, outdim, img_path)

        alphas = [0.6, 0.8, 1.0, 1.5] 
        image_column = []

        if (z_optimize):
            print("Not implemented") 

        # Inverted image by direct forward pass of the StyleGAN generator 
        out_z_img = decode_forward(nets, outdim, z)
        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

        # Creating a column by appending the original image for particular strength of att. 
        image_column.append(save_out_z_img)

        for alpha in alphas:
            # print("using alpha value of:", alpha)
            zT = z + alpha*avg_latent_dir 

            # Transformed image by using the modified latent code 
            out_zT_img = decode_forward(nets, outdim, zT) 
            save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

            image_column.append(save_out_zT_img)

        image_column = np.vstack(image_column)
        image_matrix.append(image_column)  
        # print("image column shape: ", image_column.shape)

    group_image = np.hstack(image_matrix)
    print("group image shape: ", group_image.shape)
    save_img = Image.fromarray(np.uint8(group_image)).convert('RGB') 
    fn = 'matrix_transform_imgs_' + att + '_' + str(alpha) + '.jpg'

    save_img_path = os.path.join(img_res_path, fn)

    print("saving image: ", save_img_path)
    save_img.save(save_img_path) 
 


# Editing image set with the saved latents [Vanilla], applying transformation learnt by pairwise imgs and non-paired images 
def edit_image_set(): 
    nets = load_nets()
    img_path_root = '../../CelebAMask-HQ/data_filtered/test500'
    img_res_path = '../../CelebAMask-HQ/data_filtered/renew/results/' 
    dirs_files_root = '../../data_files/estimated_dirs/'  

    img_names = [img for img in os.listdir(img_path_root)]
    print("Editing images:", img_names)
    print("n image: ", len(img_names))

    atts_list = ['bang', 'eye_g', 'smile', 'bald', 'hat', 'pose']      
    latent_paths = ['latent_db_dir_id_11_bang.npy', 'latent_db_dir_id_17_eye_g.npy', 'latent_db_dir_id_18_smile.npy', 'latent_db_dir_id_20_bald.npy', 
                   'latent_db_dir_id_20_hat.npy', 'latent_db_dir_id_14_pose.npy'] 
    alphas = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    n_pairs = 5 
    image_res_paths = [os.path.join(img_res_path, att + '_pairs_' + str(n_pairs)) for att in atts_list] 

    # Creating folders for results for each attribute 
    for im_res_pt in image_res_paths:
        if (not os.path.exists(im_res_pt)):
            os.mkdir(im_res_pt)

    latent_paths = [os.path.join(dirs_files_root, lp) for lp in latent_paths]

    # Number of images to be processed 
    n = 10
    print("Editing {} images".format(n))   

    # Saving the image edits for a set of image and all the set of attributes 
    for i in range(0, n): 
        for j in range(0, len(atts_list)):  
            att = atts_list[j]
            print("Editing Image for {} attribute".format(att))
            img_res_path_att = image_res_paths[j] # os.path.join(img_res_path, att) 
            # Taking the ith image and the jth latent attribute code for creating the edit image  
            edit_image(nets, img_path_root, img_res_path_att, img_names[i], att, latent_paths[j], alphas[j], 'pos', n_pairs, z_optimize=True)  


if __name__ == "__main__":          
  print("running main ...")
  print("editing image dirs ...")  
  edit_image_set()        
  
  