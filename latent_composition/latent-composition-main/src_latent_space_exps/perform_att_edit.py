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


# This function performs normalized the latent code differences to unit lengths 


# This function will parse the saved pairwise latent direction from a dictionary file and processed them. 
# In the processing first all the pairwise latent directions for any attribute style is normalized to unit length and then
# averaged/SVD is compute to find the dominant variation direction of the latent directions for each style for the given attribute. 
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
        # avg_dir = np.mean(normed_latents, axis=0)

        # Instead of taking averaging we are performing SVD to find the dominant variation of the direction 
        avg_dir = estimate_dominant_dir(normed_latents, 'SVD')

        # print("SVD dominant direction shape: ", avg_dir.shape)

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

# This function finds the latent direction that joins the normal_direction to every other direction and return a set of basis vectors. 
def estimate_deviation_from_normal(scaled_latents, normal_direction):
    basis_latents = [scl - normal_direction for scl in scaled_latents]
    return basis_latents 

# Check for planarity of the vectors forming the basis of the space of attributes 
def check_planarity(basis_vectors, normal_vector): 
    dots = []
    for i in range(0, len(basis_vectors)):
        dot_p = np.dot(basis_vectors[i].flatten(), normal_vector.flatten())
        dots.append(dot_p)

    print("Checking the dot products here: ", dots ) 

# This function will create a basis vector in a more meaningful way by first projecting all the latents onto the plane perpendicular to the SVD direction
# And then estimating the basis directions by taking difference for each of the direction from the SVD direction 
def create_basis_latents_advanced(latent_db_processed):  
    print("Creating basis by the defined formulation ... ")
    print("processed latent db shape: ", len(latent_db_processed))  

    normal_direction = estimate_dominant_dir(np.array(latent_db_processed), 'SVD') 
    print("normal direction shape: ", normal_direction.shape)
    normal_flatten = normal_direction.flatten() 

    # This scalling factor will make the vectors lie in a plane formed by tangent of the unit sphere. 
    scaled_latents = []

    # latents: Shape = n_pairs x 18 x 512 
    for i in range(0, len(latent_db_processed)): 
        curr_latent = latent_db_processed[i]
        latent_flatten = curr_latent.flatten() # flatten into a single array of 18 x 512 size
        cos_theta = np.dot(normal_flatten, latent_flatten) # As both of them are unit vectors 
        # |v_avg| /|p1| = cos_theta  => 1 / |p1| = cos_theta as |v_avg| = 1
        scale_factor = 1 / cos_theta
        # print("scale factor: ", scale_factor)        
        latent_scaled = curr_latent * scale_factor 

        # print("scaled latent length: ", np.linalg.norm(latent_scaled))

        scaled_latents.append(latent_scaled)

    basis_latents = estimate_deviation_from_normal(scaled_latents, normal_direction) 

    return basis_latents, normal_direction  

# Editing the image given the image path, the name of attribute to be edited and the latent db file path and number of images used for averaging. 
def edit_image_interpolate_atts(nets, img_path, img_idx, img_transform_path, att, latent_db_path, n_pairs, n_transforms, alpha, variation):    
    n_atts = 10
    fixed_scalar = 20
    outdim = 1024
    img_save_size = 256 
    latent_db_processed = parse_latent_db(latent_db_path, n_pairs)  

    n_atts = len(latent_db_processed)
    print("number of attribute styles in input: ", n_atts)

    # To obtain the basis latent vectors of the attribute style space, given the set of processed latent codes normalized followed by averaged. 
    # basis_latents_avg = create_basis_latents(latent_db_processed)
    # To compute the dc shift, we will average out all the directions for each attribute style 
    # dc_shift = np.array(latent_db_processed).mean(axis=0) 
    # print("dc shift vector shape: ", dc_shift.shape) 

    # Both the basis vectors and the dc shift will come from the create basis function 
    basis_latents, dc_shift = create_basis_latents_advanced(latent_db_processed)  

    # checking whether the basis are in the plane or not 
    check_planarity(basis_latents, dc_shift) 

    for id in range(0, n_transforms): 
        basis_coeffs = [round(np.random.uniform(-variation, variation),2) for i in range(0,n_atts-1)]
        avg_latent_dir = compute_interpolate_dir(basis_latents, basis_coeffs)
        
        avg_latent_dir_shifted = avg_latent_dir + dc_shift
        latent_dir_tensor = torch.from_numpy(avg_latent_dir_shifted).cuda()

        z, save_src_img = encode_forward(nets, outdim, img_save_size, img_path)
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
def edit_image_interpolate_atts_group(nets, img_path, img_idx, img_transform_path, att, latent_db_path, n_pairs, n_transforms, alphas, variation): 
    n_atts = 10
    fixed_scalar = 20
    outdim = 1024
    img_save_size = 256 
    latent_db_processed = parse_latent_db(latent_db_path, n_pairs)         

    n_atts = len(latent_db_processed)
    print("number of attribute styles in input: ", n_atts) 
    print("Editing attribute: ", att)

    # To obtain the basis latent vectors of the attribute style space, given the set of processed latent codes normalized followed by averaged. 
    # basis_latents_avg = create_basis_latents(latent_db_processed)
    # To compute the dc shift, we will average out all the directions for each attribute style 
    # dc_shift = np.array(latent_db_processed).mean(axis=0) 
    # print("dc shift vector shape: ", dc_shift.shape) 

    # Both the basis vectors and the dc shift will come from the create basis function 
    basis_latents, dc_shift = create_basis_latents_advanced(latent_db_processed)  

    # checking whether the basis are in the plane or not 
    check_planarity(basis_latents, dc_shift) 
    # print("dc shift vector shape: ", dc_shift.shape, dc_shift.min(), dc_shift.max())   

    image_matrix = []
    for id in range(0, n_transforms):
        # Source image decode latent code 
        z, save_src_img = encode_forward(nets, outdim, img_save_size, img_path)
        out_z_img = decode_forward(nets, outdim, z)
        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 

        # Iterating over image column 
        image_column = [save_src_img, save_out_z_img]

        # Iterating over all the values of alphas  
        for alpha in alphas:
            basis_coeffs = [round(np.random.uniform(-variation, variation),2) for i in range(0,n_atts-1)] 
            avg_latent_dir = compute_interpolate_dir(basis_latents, basis_coeffs) 

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

# This function will create a gif for the attribute style editing operation
def create_gif(nets, img_path, img_idx, img_transform_path, att, latent_db_path, n_pairs, n_key_frames, variation, alpha):
    print("Creating gif .... ")
    n_atts = 10
    fixed_scalar = 20
    outdim = 1024
    img_save_size = 512
    latent_db_processed = parse_latent_db(latent_db_path, n_pairs)

    n_atts = len(latent_db_processed)
    print("number of attribute styles for input: ", n_atts)

    # To obtain the basis latent vectors of the attribute style space, given the set of processed latent codes normalized followed by averaged. 
    # basis_latents_avg = create_basis_latents(latent_db_processed)
    # To compute the dc shift, we will average out all the directions for each attribute style 
    # dc_shift = np.array(latent_db_processed).mean(axis=0) 
    # print("dc shift vector shape: ", dc_shift.shape) 

    # Both the basis vectors and the dc shift will come from the create basis function 
    basis_latents, dc_shift = create_basis_latents_advanced(latent_db_processed)  

    # checking whether the basis are in the plane or not 
    check_planarity(basis_latents, dc_shift) 
    # print("dc shift vector shape: ", dc_shift.shape, dc_shift.min(), dc_shift.max()) 

    # Generating keyframes for interpolation 
    key_weights = []
    for id in range(0, n_key_frames):
        basis_coeffs = [round(np.random.uniform(-variation, variation), 2) for i in range(0,n_atts-1)]
        key_weights.append(basis_coeffs) 

    # Creating a circular loop to make the visual look continuous 
    key_weights.append(key_weights[0])

    z, save_src_img = encode_forward(nets, outdim, img_save_size, img_path)
    out_z_img = decode_forward(nets, outdim, z)
    out_z_hr_img = renormalize.as_image(out_z_img[0]).resize((outdim, outdim), Image.LANCZOS)


    quantize = 0.1 
    accumulated_images = [] 
    accumulated_hr_images = [out_z_hr_img] 

    for i in range(0, len(key_weights)-1):
        for t in range(0, int(1/quantize)):
            # Interpolating between the keyframe weights 
            weights_array = (1-quantize*t) * np.array(key_weights[i]) + quantize*t * np.array(key_weights[i+1])
            weights = list(weights_array)
            avg_latent_dir = compute_interpolate_dir(basis_latents, weights)

            avg_latent_dir_shifted = avg_latent_dir + dc_shift
            
            # Normalizing the latent direction before using it for editing as it can have large variation of values 
            avg_latent_dir_shifted = avg_latent_dir_shifted / np.linalg.norm(avg_latent_dir_shifted)
            # print("length of the final vector used for editing: ", np.linalg.norm(avg_latent_dir_shifted))  

            latent_dir_tensor = torch.from_numpy(avg_latent_dir_shifted).cuda()

            sample_alpha = np.random.uniform(alpha,alpha + 0.3)
            zT = z + sample_alpha * fixed_scalar * latent_dir_tensor

            out_zT_img = decode_forward(nets, outdim, zT)
            save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
            save_out_hr_img = renormalize.as_image(out_zT_img[0]).resize((outdim, outdim), Image.LANCZOS) 
            processed_img = Image.fromarray(np.uint8(save_out_zT_img)).convert('RGB')

            accumulated_images.append(processed_img)
            accumulated_hr_images.append(save_out_hr_img) 

    fn = img_idx[:-4] + '_transformed_gif_' + att + '_interpolation_alpha' + str(alpha) + '_variation' + str(variation) + '.gif'
    save_img_path = os.path.join(img_transform_path, 'gifs', fn)
    # Saving the animation gif at the destination location 
    accumulated_images[0].save(save_img_path, save_all = True, append_images=accumulated_images)      
    print("Saved the processed animation at: {}".format(save_img_path), "Enjoy!")

    # Saving all the variations of attribute styles into separate files. 
    fp_src = os.path.join(img_transform_path, img_idx[:-4])
    if (not os.path.exists(fp_src)):
        os.mkdir(fp_src)

    
    for id in range(0, len(accumulated_hr_images)):
        img = accumulated_hr_images[id]
        fn = os.path.join(fp_src, str(id) + '.jpg')
        # Saving the processed file as a new separate file 
        processed_img = Image.fromarray(np.uint8(img)).convert('RGB')
        processed_img.save(fn)
    print("Saved individual files for transformations")
    

# This function has the main functionality to read and image and create attribute style transformations over it. 
# Once we define the attributes to be considered and the paths for their latent codes then edit image can be called for
# each of the attribute separately. 
def edit_image_set_interpolate_atts():
    nets = load_nets()
    img_path_root = '../../CelebAMask-HQ/data_filtered/test500'
    img_transform_path_root = '../../CelebAMask-HQ/data_filtered/renew/results_display/attr_style' 
    latent_path_root = '../../data_files/estimated_att_styles_filt'

    # Specifieng paths for the latent db for processesing 
    att_list = ['eye_g_style', 'hair_style']
    latent_db_list = ['latent_style_att_dir_db_eye_g_style.csv', 'latent_style_att_dir_db_hair_style.csv']  
    latent_db_paths = [os.path.join(latent_path_root, ln) for ln in latent_db_list] 
    

    img_idxs = [img for img in os.listdir(img_path_root) if(img[-4:] == '.jpg')]
    # Randomly shuffle image_idxs: | Not using random shuffle for now 
    import random
    img_idxs = random.sample(img_idxs, len(img_idxs))  

    img_paths = [os.path.join(img_path_root, img_id) for img_id in img_idxs]

    # Number of images to be processed, number of pairs to be used for computation and the required edit strength 
    n = 50
    n_pairs = 5  
    n_transforms = 16
    n_key_frames = 10 
    edit_alpha = 0.6 * 0.4 # Based on the latent manipulation on a fixed set of layers 
    gif_alphas = [0.6 * 0.6, 0.6 * 0.8] # Based on the manipulation on a fixed set of layers 
    alphas = [0.2, 0.3, 0.5, 0.8]  # Passing various values of alpha to generate attribute varaitions with different strengths 
    variations = [0.8, 1.5]  # Passing different level of variation to be used for different attributes 
    print("Editing {} images".format(n)) 

    # Saving results with projection logic used for basis estimation 
    img_transform_paths = [os.path.join(img_transform_path_root, att + '_' + str(n_pairs) + '_proj') for att in att_list]

    for i in range(0, n):  
        for att_j in range(0, 2): # Currently running just for the hair attribute 
            # edit_image_interpolate_atts(nets, img_paths[i], img_idxs[i], img_transform_paths[att_j], att_list[att_j], latent_db_paths[att_j], n_pairs, n_transforms, edit_alpha, variations[att_j])
            # edit_image_interpolate_atts_group(nets, img_paths[i], img_idxs[i], img_transform_paths[att_j], att_list[att_j], 
            #                                   latent_db_paths[att_j], n_pairs, n_transforms, alphas, variations[att_j])
            create_gif(nets, img_paths[i], img_idxs[i], img_transform_paths[att_j], att_list[att_j], latent_db_paths[att_j], n_pairs, n_key_frames, variations[att_j], gif_alphas[att_j])

if __name__ == "__main__":   
  print("running main ...")
  print("editing image dirs ...")  

  print("editing image with interpolation of attributes") 
  edit_image_set_interpolate_atts()   
