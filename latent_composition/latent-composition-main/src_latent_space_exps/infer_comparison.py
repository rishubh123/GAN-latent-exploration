"""
Date: 19/01/2022
This module will just forward pass the latent codes extracted with the required transformation applied to perfrom desired edit for comparison.
""" 


import os 
import numpy as np
import torch 
from stylegan_utils import encode_forward, decode_forward, load_image_tensor, load_nets 
from utils import renormalize, inversions 
from PIL import Image   
import pickle 

# sys.path.append('../')   

# Helper for generting our latents, This function will fuse the latent directions by different fusion type
def estimate_dominant_dir(latents, fusion): 
    # print("Computing SVD to find the dominant direction.")
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


# Helper function to estimate latents for our method, This function performs normalized the latent code differences to unit length 
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


# This function will create a forward pass and returns the image corresponding to the given latent code 
def infer_model(nets, outdim, latent):
    img = decode_forward(nets, outdim, latent)
    return img 

# This function will compute the transforms of given image using the latent directions estimated by interfaceGAN method 
def compute_transform_if(nets, outdim, z, latent_path_if, attr_list, attr_strengths, seq):
    print("Editing with interface model")
    n_attr = len(attr_list) 
    latent_name = ['smile.npy', 'angle_horizontal.npy', 'age.npy']
    latent_paths = [os.path.join(latent_path_if, ln) for ln in latent_name]
    latents = [np.load(lp) for lp in latent_paths]

    # Transformed z vectors for each of the attribute one at a time
    zTs = [z]
    for id in range(n_attr):
        z_new = z + attr_strengths[id] * latents[id]
        # If we are performing sequential edits then the z will change to z_new 
        if (seq):
            z = z_new
        zTs.append(z_new) 

    out_image_stack = []
    for id in range(n_attr + 1):
        z_current = zTs[id]
        latent = torch.from_numpy(z_current)
        latent_torch = latent.type(torch.FloatTensor).cuda()

        out_img = infer_model(nets, outdim, latent_torch)
        img_norm = renormalize.as_image(out_img[0]).resize((outdim, outdim), Image.LANCZOS) 
        out_image_stack.append(img_norm) 

    print("Returning image stack from interfaceGAN of shape: ", len(out_image_stack))
    out_image_stack = np.hstack(out_image_stack)
    return out_image_stack

# This function will compute the attribute editing with our method and returns a stack of images. 
def compute_transform_our(nets, outdim, z, latent_path_ours, attr_list, attr_strengths, seq):
    print("Editing with our method")
    n_attr = len(attr_list)
    n_pairs = 5
    latent_names = ['latent_db_dir_id_18_smile.npy', 'latent_db_dir_id_14_pose.npy', 'latent_db_dir_id_12_age_80.npy'] 
    latent_paths = [os.path.join(latent_path_ours, ln) for ln in latent_names]
    latent_dbs = [np.load(lp) for lp in latent_paths] 

    latents = []
    for l in range(n_attr):
        latent_db_dir = latent_dbs[l]
        latent_dir_db_norm = normalize_dirs(latent_db_dir)               
        # Filtering out the first set of 5 latents for processing
        latent_dir_db_norm = latent_dir_db_norm[:n_pairs, ...]  
        # Using SVD to find the dominant direction 
        avg_latent_dir = estimate_dominant_dir(latent_dir_db_norm, 'SVD')   
        latents.append(avg_latent_dir) 

    # Transformed z vectors for each of the attribute one at a time
    zTs = [z]
    for id in range(n_attr): 
        z_new = z + attr_strengths[id] * latents[id]
        # If we are performing sequential edits then the z will change to z_new 
        if (seq):
            z = z_new
        zTs.append(z_new) 

    out_image_stack = []
    for id in range(n_attr + 1):
        z_current = zTs[id]
        latent = torch.from_numpy(z_current)
        latent_torch = latent.type(torch.FloatTensor).cuda()

        out_img = infer_model(nets, outdim, latent_torch)
        img_norm = renormalize.as_image(out_img[0]).resize((outdim, outdim), Image.LANCZOS) 
        out_image_stack.append(img_norm) 

    print("Returning image stack from interfaceGAN of shape: ", len(out_image_stack))
    out_image_stack = np.hstack(out_image_stack)
    return out_image_stack


# Tjis function performs a forward pass for the estimated latents from the styleflow approach 
def compute_transform_sf(nets, outdim, z, z_transforms, id, attr_list):
    print("Editing with StyleFlow ...")
    n_attr = len(attr_list)
    
    z_transform = z_transforms[id, ...] 
    zTs = [z]
    for id in range(n_attr):
        z_new = z_transform[id, ...]
        zTs.append(z_new)

    out_image_stack = []
    for id in range(n_attr + 1):
        z_current = zTs[id]
        latent = torch.from_numpy(z_current)
        latent_torch = latent.type(torch.FloatTensor).cuda()

        out_img = infer_model(nets, outdim, latent_torch)
        img_norm = renormalize.as_image(out_img[0]).resize((outdim, outdim), Image.LANCZOS) 
        out_image_stack.append(img_norm) 


    print("Returning image stack from interfaceGAN of shape: ", len(out_image_stack))
    out_image_stack = np.hstack(out_image_stack)
    return out_image_stack



# This function will inflate the latent code of 512 dimensions to match the dimensions of w+ space which is 18x512
def convert_to_wplus(Z_comp):
    # Array where W+ latent directions will be save by replicating copies of W latent code 
    Z_reshaped = np.zeros((80, 18, 512)) # As we have 80 componetns fixed from GanSpace algorithm 
    for i in range(0, 80):
        Z_list = [Z_comp[i, :, :] for k in range(0,18)]
        Z_stack = np.vstack(Z_list)

        # print("Z stack shape: ", Z_stack.shape)
        Z_reshaped[i, : , :] = Z_stack
    
    return Z_reshaped



# This function will compute the transforms of given image using the latent directions estimated by GanSpace and some are filtered by us 
def compute_transform_gs(nets, outdim, z, latent_path_gs, attr_list, attr_strength, seq):
    print("Editing with Ganspace")
    n_attr = len(attr_list)
    print("attribute strengths: ", attr_strength) 
    latent_name = 'stylegan2-ffhq_style_ipca_c80_n1000000_w.npz'
    latent_path = os.path.join(latent_path_gs, latent_name)

    latent_loaded = np.load(latent_path, allow_pickle=False) 
    Z_comp = latent_loaded['lat_comp']
    Z_mean = latent_loaded['lat_mean']

    latents_all = convert_to_wplus(Z_comp)
    latents = [latents_all[46,:,:], latents_all[1,:,:], latents_all[34,:,:]] 

    zTs = [z]
    for id in range(n_attr):
        z_new = z + attr_strength[id] * latents[id]
        # If we are creating sequential edits then the z will be updated as z_new 
        if (seq): 
            z = z_new 
        zTs.append(z_new)

    out_image_stack = []
    for id in range(n_attr + 1):
        z_current = zTs[id]
        latent = torch.from_numpy(z_current)
        latent_torch = latent.type(torch.FloatTensor).cuda()

        out_img = infer_model(nets, outdim, latent_torch)
        img_norm = renormalize.as_image(out_img[0]).resize((outdim, outdim), Image.LANCZOS) 
        out_image_stack.append(img_norm) 

    print("Returning the image stack from GanSpace of shape: ", len(out_image_stack)) 
    out_image_stack = np.hstack(out_image_stack)
    return out_image_stack




# This module will create the latent space transformation for a set of synthetic images provided with the latent code from styleFlow testing set 
def compare_main():
    style_flow_imgs = '../../../../StyleFlow/StyleFlow/data/sg2latents.pickle'
    dst_path = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/compare_matrix/'
    loaded_src_latents = pickle.load(open(style_flow_imgs, "rb")) 
    src_img_latents = loaded_src_latents['Latent']  

    print("latent for Gan Space dimentions ", len(src_img_latents))

    attr_list = ['smile', 'pose', 'age']
    attr_strength_if = [2.5, 5.0, -5.0] # attribute strength for latent transformation using interfaceGAN
    attr_strength_gs = [-1.0, 1.0, 2.0] # attribute strength for latent edit using GanSpace directions 
    attr_strength_our = [-0.4, 0.4, 0.56] # attribute strength for latent edits from our proposed method directions 

    gs_fixed_strength = 3
    attr_strength_gs = [gs_fixed_strength * alpha for alpha in attr_strength_gs] 

    fixed_strength = 10 # Fixed multiplier for our method to create visible transformation 
    attr_strength_our = [fixed_strength * alpha for alpha in attr_strength_our]   
    # For styleflow model the attribute strength will be already incorporated in the transformed latents which we will directly consume. 

    latent_path_if = '../../data_files/comparison_files/InterfaceGAN/latent_directions'
    latent_path_gs = '../../data_files/comparison_files/GanSpace/latent_directions'
    tform_latents_sf ='../../../../StyleFlow/StyleFlow/results/individual_latents'
    latent_path_ours = '../../data_files/estimated_dirs_filt/'

    n = 21
    seq = False 
    nets = load_nets()
    outdim = 1024

    # Traversing over all the images and calling the edits to obtain transformed images. 
    for id in range(0,n):
        img_src_latent = src_img_latents[id]

        stack_ifg = compute_transform_if(nets, outdim, img_src_latent, latent_path_if, attr_list, attr_strength_if, seq)
        stack_gs = compute_transform_gs(nets, outdim, img_src_latent, latent_path_gs, attr_list, attr_strength_gs, seq)
        stack_our = compute_transform_our(nets, outdim, img_src_latent, latent_path_ours, attr_list, attr_strength_our, seq)
        # stack_sf = compute_transform_sf(nets, outdim, img_src_latent, tform_latents_sf, id, attr_list)

        combined_image_stack = np.vstack([stack_ifg, stack_gs, stack_our])

        img_save_name = str(id) + '_compare_mat_' + str(attr_list) + '_seq_' + str(seq) + '.jpg'
        img_save_path = os.path.join(dst_path, img_save_name)

        save_img = Image.fromarray(np.array(combined_image_stack, np.uint8)).convert('RGB') 
        print("Saving image stack at: ", img_save_path) 

        save_img.save(img_save_path) 


if __name__ == "__main__":
    compare_main()
