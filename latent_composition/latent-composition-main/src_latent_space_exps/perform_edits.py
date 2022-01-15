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

# This function will fuse the latent directions by different fusion type
def estimate_dominant_dir(latents, fusion): 
    print("Computing SVD to find the dominant direction.")
    if (fusion == 'SVD'):
        data_matrix = []
        # latents: Shape = n_pairs x 18 x 512 
        for i in range(0, latents.shape[0]): 
            latent_flatten = latents[i].flatten() # flatten into a single array of 18 x 512 size
            data_matrix.append(latent_flatten)

        # Creating a single matrix for performing SVD over normalizing factor 
        data_matrix = np.array(data_matrix) 
        
        u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)  
        # print("eigen values: ", s)
        dominant_dir = vh[0,:].reshape(18, 512)

        magnitude = np.linalg.norm(dominant_dir) 
        dominant_dir = -dominant_dir / magnitude   
 
        return dominant_dir   


# Function to compute the dot product between two latent vectors by flatenning and taking a dot product
def compute_dot_product(latent1, latent2):
    dot_value = latent1.dot(latent2)
    return dot_value 

# This function will orthogonalize the all the attributes to have only orthogonal direction to other attributes
def orthogonalize(latent_list):
    orthogonal_latents = []
    n = len(latent_list)

    # Iterating over all the latents to be altered
    for i in range(0,n):
        current_latent = latent_list[i]
        # Taking dot products with all the other latent directions and editing the output image 
        for j in range(0,n):
            if (j != i):
                shift = latent_list[i].flatten().dot(latent_list[j].flatten().T) * latent_list[j]
                # shift_size = torch.norm(shift)
                current_latent -= shift 
                current_size = torch.norm(current_latent)
            
        # Normalizing the latent code to match the curent latent size to match one 
        current_latent = current_latent / current_size
        orthogonal_latents.append(current_latent)   

    return orthogonal_latents  

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
    # avg_latent_dir = latent_dir_db_norm.mean(axis=0)

    # Using SVD to find the dominant direction 
    avg_latent_dir = estimate_dominant_dir(latent_dir_db_norm, 'SVD')    
    
    # Using SVD we will try to estimate the SVD for getting the dominant direction
    avg_latent_dir = torch.from_numpy(avg_latent_dir).cuda()    

    # If this flag is present then the latent code of the input image will be optimized first then the latent code transformation will be performed. 
    if (z_optimize):
        source_im = load_image_tensor(img_path, outdim) 

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
        z, save_src_img = encode_forward(nets, outdim, img_save_size, img_path)

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
    inversion_img_hr = renormalize.as_image(out_z_img[0]).resize((outdim, outdim), Image.LANCZOS)
    out_img_hr = renormalize.as_image(out_zT_img[0]).resize((outdim, outdim), Image.LANCZOS) 

    save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
    save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
    
    # Creating a combined stack image that will be saved for analysis and visualization 
    combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
    save_img = Image.fromarray(np.array(combined_display_image, np.uint8)).convert('RGB') 

    fn_combined = img_name[:-4] + '_stack_' + att + '_' + dir + '_alpha_' + str(alpha) + '_' + str(z_optimize) + '.jpg'  # removing .jpg
    fn_res = img_name[:-4] + '_transformed_' + att + '_' + dir + '_alpha_' + str(alpha) + '_'+ str(z_optimize) + '.jpg'  
    fn_inversion = img_name[:-4] + '_inversion.jpg' 

    # Saving the transformed images and the stack of transformed and original image 
    fn_combined_save_path = os.path.join(img_res_path, fn_combined)
    fn_res_save_path = os.path.join(img_res_path, fn_res) 
    fn_inversion_path = os.path.join(img_res_path, fn_inversion)         

    print("saving image stack at:", fn_combined_save_path)
    print("saving inversion:", fn_inversion_path)
    save_img.save(fn_combined_save_path) 
    out_img_hr.save(fn_res_save_path) 
    inversion_img_hr.save(fn_inversion_path)
  
# Perform multiple sequential attribute edits to evaluate robustness, we use a list of alphas, basically chaning the alpha for any of the attribute to see visual results 
def edit_image_sequentially(nets, img_src_root, img_res_path, img_name, att_list, latent_db_paths, alphas, dir, n_pairs, ognl, z_optimize): 
    outdim = 1024
    img_save_size = 256 # Image resolution at which the inversion and transformed image stack will be dumbed for visualization 
    fixed_scalar = 10 # Fixed scalar to map the unit normalized vectors back to a meaningful value 
    img_path = os.path.join(img_src_root, img_name)

    avg_latent_dirs = []
    # Iterating over all the attributes and preprocessing the latents for performing sequential edit operations 
    for ldb in latent_db_paths:
        # print("loading att latent dir file: ", ldb)  
        latent_dir_db = np.load(ldb)
        latent_dir_db_norm = normalize_dirs(latent_dir_db)            

        # Filtering out the first set of 5 latents for processing
        latent_dir_db_norm = latent_dir_db_norm[:n_pairs, ...]

        # Average latent direction to be used for image editing 
        # avg_latent_dir = latent_dir_db_norm.mean(axis=0)

        # Using SVD to find the dominant direction 
        avg_latent_dir = estimate_dominant_dir(latent_dir_db_norm, 'SVD')    
        avg_latent_dir = torch.from_numpy(avg_latent_dir).cuda()     
        avg_latent_dirs.append(avg_latent_dir) 

    # Orthogonalize latents
    if (ognl):
        print("Editing with orthogonalization in the sequential edit vectors ... ")
        avg_latent_dirs = orthogonalize(avg_latent_dirs) 


    # If this flag is present then the latent code of the input image will be optimized first then the latent code transformation will be performed. 
    if (z_optimize):
        source_im = load_image_tensor(img_path, outdim) 

        # Note: There is some code issue for inversion optimization, have to recheck and rectify
        # Performing the latent optimization to estimate the identity preserved direction 
        # checkpoint_dict, opt_losses = inversions.optimize_latent_for_id(nets, source_im, avg_latent_dir, att_stength) 
        checkpoint_dict, opt_losses = inversions.invert_lbfgs(nets, source_im, mask=None, lambda_f=0.25, lambda_l=0.5, num_steps=3000, initial_latent=None)
        
        # Image to be saved in small size 
        save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
        out_z_img = checkpoint_dict['current_x'].detach().clone() 
        z = checkpoint_dict['current_z'].detach().clone()

    else:
        z, save_src_img = encode_forward(nets, outdim, img_save_size, img_path)

    # Inversion image which will be saved once in the stack of output images 
    out_z_img = decode_forward(nets, outdim, z)

    output_images = [out_z_img] 
    for id in range(0, len(att_list)):
        att = att_list[id]
        avg_latent_dir = avg_latent_dirs[id]

        # Changing the z vector for edit based on the edit strength and the fixed scalar values 
        # Transfromation of z for changing the latent in positive direction if dir is positive  
        if (dir == 'pos'):
            zT = z + alphas[id]*fixed_scalar*avg_latent_dir 

        # IF the direction is negative then we will subtract the direction from the current latent code for transformation 
        if (dir == 'neg'):
            zT = z - alphas[id]*fixed_scalar*avg_latent_dir
        
        out_zT_img = decode_forward(nets, outdim, zT)
        output_images.append(out_zT_img) 

        z = zT 
    
    # Mapping the outpus generated to an image
    output_images_norm = [save_src_img]
    for img in output_images:
        img_norm = renormalize.as_image(img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)  
        output_images_norm.append(img_norm)

    # Taking the last image as the HR output for the final transformed image 
    out_img_hr = renormalize.as_image(output_images[-1][0]).resize((outdim, outdim), Image.LANCZOS) 
    
    # Creating a combined stack image that will be saved for analysis and visualization 
    combined_display_image = np.hstack(output_images_norm) 
    save_img = Image.fromarray(np.array(combined_display_image, np.uint8)).convert('RGB')

    fn_combined = img_name[:-4] + '_sequential_stack_' + str(att_list) + '_' + dir  + '_' + str(z_optimize) + '_' + '_ognl_' + str(ognl) + '.jpg'  # removing .jpg
    fn_res = img_name[:-4] + '_sequential_transformed_' + str(att_list) + '_' + dir + '_'+ str(z_optimize) + '_' + '_ognl_' + str(ognl) + '.jpg'  

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
        z, save_src_img = encode_forward(nets, outdim, img_save_size, img_path) 

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
    img_res_path = '../../CelebAMask-HQ/data_filtered/renew/results_display/' 
    dirs_files_root = '../../data_files/estimated_dirs_filt/'  
    img_res_seq_path = os.path.join(img_res_path, 'sequential_svd_filt')  

    img_names = [img for img in os.listdir(img_path_root)]
    # print("Editing images:", img_names)
    print("n image: ", len(img_names))

    # atts_list = ['pose', 'smile', 'bald', 'eye_g', 'bang', 'hat'] 
    atts_list = ['bang', 'eye_g', 'smile', 'bald', 'hat', 'pose', 'age_60_', 'age_80_' 'beard']       
    latent_paths = ['latent_db_dir_id_11_bang.npy', 'latent_db_dir_id_17_eye_g.npy', 'latent_db_dir_id_18_smile.npy', 'latent_db_dir_id_20_bald.npy', 
                   'latent_db_dir_id_20_hat.npy', 'latent_db_dir_id_14_pose.npy', 'latent_db_dir_id_12_age_60.npy', 'latent_db_dir_id_12_age_80.npy', 'latent_db_dir_id_8_beard.npy'] 
    alphas = [1.0, 1.2, 0.5, 1.0, 2.0, 1.0, 1.2, 1.2, 0.8] 
    alphas_filt = [0.4*alphas[i] for i in range(0, len(alphas))] 
    # alphas_filt = [0.2*alphas[i] for i in range(0, len(alphas))]
    
    n_pairs = 5 
    # Number of images to be processed 
    n = 100  
    print("Editing {} images".format(n))   

    img_names = img_names[:n] 
    # Instead of saving transformations for each attribute in a specific folder to them, we will create folder for image names and save all the transformations separately there. 
    # image_res_paths = [os.path.join(img_res_path, att + '_pairs_' + str(n_pairs)) for att in atts_list] 
    image_res_paths = [os.path.join(img_res_path, imn[:-4]) for imn in img_names] 

    # Working with only age and beard attribute 
    # alphas = alphas[-2:]
    # atts_list = atts_list[-2:]
    # latent_paths = latent_paths[-2:]
    # image_res_paths = image_res_paths[-2:]

    # Creating folders for results for each attribute 
    for im_res_pt in image_res_paths:
        if (not os.path.exists(im_res_pt)):
            os.mkdir(im_res_pt)

    latent_paths = [os.path.join(dirs_files_root, lp) for lp in latent_paths] 

    # Performing edits of one attribute at a time
    # Saving the image edits for a set of image and all the set of attributes 
    for i in range(0, n): 
        for j in range(0, len(atts_list)):  
            att = atts_list[j]
            # Taking the ith image and the jth latent attribute code for creating the edit image  
            edit_image(nets, img_path_root, image_res_paths[i], img_names[i], att, latent_paths[j], alphas_filt[j], 'pos', n_pairs, z_optimize=False)     


    ## -----------------------------------------------## 
    # Performing sequential edits for robustness 
    alphas = [1.0, 0.5, 1.5, 2.0]   
    alphas_filt = [0.4*alphas[i] for i in range(0, len(alphas))]     
    atts_list = ['pose', 'smile', 'age_60_', 'eye_g'] 

    latent_paths = ['latent_db_dir_id_14_pose.npy', 'latent_db_dir_id_18_smile.npy', 
                    'latent_db_dir_id_12_age_60.npy', 'latent_db_dir_id_17_eye_g.npy']      
    latent_paths = [os.path.join(dirs_files_root, lp) for lp in latent_paths] 



    # Editing only the last two attributes which are current here.   
    # alphas = alphas[-2:]
    # atts_list = atts_list[-2:]
    # latent_paths = latent_paths[-2:] 
   
    """
    print("Performing sequential Edits on the image ... ")
    # Performing edit of the image by sequential attribute editing to evaluate the robustness of editing operations 
    for i in range(0, n):
        edit_image_sequentially(nets, img_path_root, img_res_seq_path, img_names[i], atts_list, latent_paths, alphas_filt, 'pos', n_pairs, ognl=True, z_optimize=False)
    """ 

if __name__ == "__main__":          
  print("running main ...")
  print("editing image dirs ...")  
  edit_image_set()        
  
  