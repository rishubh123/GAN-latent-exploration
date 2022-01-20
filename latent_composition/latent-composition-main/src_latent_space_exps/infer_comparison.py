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
    latent_name = ['smile.npy', 'angle_horizontal.npy', 'age.npy', 'glasses.npy']  
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
    n_pairs = 10
    print("Editing with 10 pair of images")
    latent_names = ['latent_db_dir_id_18_smile.npy', 'latent_db_dir_id_14_pose.npy', 'latent_db_dir_id_12_age_80.npy', 'latent_db_dir_id_17_eye_g.npy'] 
    latent_paths = [os.path.join(latent_path_ours, ln) for ln in latent_names]
    latent_dbs = [np.load(lp) for lp in latent_paths] 

    latents = []
    for l in range(n_attr):
        latent_db_dir = latent_dbs[l]
        latent_dir_db_norm = normalize_dirs(latent_db_dir)               
        # Filtering out the first set of 5/10 latents for processing
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
def compute_transform_sf(nets, outdim, z, latent_path_sf, id, attr_list):
    print("Editing with StyleFlow ...")
    edited_latents_path = os.path.join(latent_path_sf, str(id) + '_orig_edits.npy')    
    z_transfs = np.load(edited_latents_path)  # Only having the required edit operations saved in the latents 

    # z_transf_order = ['Pose', 'Expression', 'Gender', 'Age', 'Eyeglasses', 'Baldness'] 
    # z_transfs = [z_transf_all[2], z_transf_all[1], z_transf_all[4], z_transf_all[5]] 
    # Selecting only that attributes that will be used for comparison 

    # print("Z loaded shape: ", z_transfs.shape)
    n_attr = len(attr_list)
    # print("n attribute: ", n_attr) 


    out_image_stack = []
    for id in range(n_attr + 1):
        z_current = z_transfs[id]
        latent = torch.from_numpy(z_current)
        latent_torch = latent.type(torch.FloatTensor).cuda()

        out_img = infer_model(nets, outdim, latent_torch)
        img_norm = renormalize.as_image(out_img[0]).resize((outdim, outdim), Image.LANCZOS) 
        out_image_stack.append(img_norm) 


    print("Returning image stack from StyleFlow of shape: ", len(out_image_stack))
    out_image_stack = np.hstack(out_image_stack)
    return out_image_stack

# This function will zero out the latent directions corresponding for each attribute based  asdflkjhjgghvafksdlfavjh vhkjfvf h fhjjf  fkjf iv fajijfklj fi klj fl fjiosa
def filter_gs_latents(attr, latent):
    print("Filtering latent for ganspace model: ", latent.shape)
    latent_modified = np.zeros((18,512))
    
    if (attr == 'Expression'):
        latent_modified[0:4, :] = np.zeros((4, 512))
        latent_modified[6:18, :] = np.zeros((12, 512))

    elif (attr == 'Pose'):
        latent_modified[4:18, :] = np.zeros((14, 512))

    elif (attr == 'Age'):
        latent_modified[0:4, :] = np.zeros((4, 512))
        latent_modified[8:18, :] = np.zeros((10, 512))
    
    elif (attr == 'Glasses'):
        latent_modified[9:18, :] = np.zeros((9, 512)) 


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
    latent_name = 'stylegan2-ffhq_style_ipca_c80_n1000000_w.npz'
    latent_path = os.path.join(latent_path_gs, latent_name)

    latent_loaded = np.load(latent_path, allow_pickle=False) 
    Z_comp = latent_loaded['lat_comp']
    Z_mean = latent_loaded['lat_mean']
    Z_std = latent_loaded['lat_stdev']

    # print("Z mean shape: ", Z_mean.shape, " Z std shape: ", Z_std.shape)


    latents_all = convert_to_wplus(Z_comp)
    # Expression, Pose, Female_old (34) - Old (15), Glasses 
    latents = [latents_all[46,:,:], latents_all[1,:,:], latents_all[15,:,:], latents_all[3,:,:]] 
    attr_list = ['Expression', 'Pose', 'Age', 'Glasses']

    # Computing the important latent layers for any particular attribute edit 
    for id in range(len(attr_list)):
        attr = attr_list[id]
        latent = latents[id]
        filter_gs_latents(attr, latent) 

    zTs = [z]
    for id in range(n_attr):

        # Different logic to perform edit by first shifting mean
        # ---------
        # w_centered = z - Z_mean    
        # w_coord = np.sum(w_centered.reshape(-1)*latents[id].reshape(-1)) / Z_std[id]
        
        # Otherwise reinforce existing
        # delta = attr_strength[id] - w_coord # offset vector

        # z_new = z + delta * latents[id]
        # ---------



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

    """
    src_test500_latents = '../../CelebAMask-HQ/data_filtered/test500_latents/' 
    src_img_latents = []
    for file_ in os.listdir(src_test500_latents):
        latent_load = np.load(os.path.join(src_test500_latents, file_))
        src_img_latents.append(latent_load)
    print("Working with images from our test set") 
    """

    print("latent for Gan Space dimentions ", len(src_img_latents))

    attr_list = ['smile', 'pose', 'age', 'eye_g']
    attr_strength_if = [3.0, 10.0, -9.0, -5.0] # attribute strength for latent transformation using interfaceGAN
    attr_strength_gs = [1.0, -0.8, -1.0, 1.0] # attribute strength for latent edit using GanSpace directions 
    attr_strength_our = [-0.3, 0.4, 0.56, 0.6] # attribute strength for latent edits from our proposed method directions 

    gs_fixed_strength = 3.0
    attr_strength_gs = [gs_fixed_strength * alpha for alpha in attr_strength_gs]  

    fixed_strength = 10 # Fixed multiplier for our method to create visible transformation 
    attr_strength_our = [fixed_strength * alpha for alpha in attr_strength_our]   
    # For styleflow model the attribute strength will be already incorporated in the transformed latents which we will directly consume. 

    latent_path_if = '../../data_files/comparison_files/InterfaceGAN/latent_directions'
    latent_path_gs = '../../data_files/comparison_files/GanSpace/latent_directions'
    tform_latents_sf ='../../../../StyleFlow/StyleFlow/results/individual_latents'
    latent_path_ours = '../../data_files/estimated_dirs_filt/'

    n = 10
    seq = True 
    nets = load_nets()
    outdim = 1024

    # Traversing over all the images and calling the edits to obtain transformed images. 
    for id in range(0,n):
        img_src_latent = src_img_latents[id]

        stack_ifg = compute_transform_if(nets, outdim, img_src_latent, latent_path_if, attr_list, attr_strength_if, seq)
        stack_gs = compute_transform_gs(nets, outdim, img_src_latent, latent_path_gs, attr_list, attr_strength_gs, seq)
        stack_sf = compute_transform_sf(nets, outdim, img_src_latent, tform_latents_sf, id, attr_list)
        stack_our = compute_transform_our(nets, outdim, img_src_latent, latent_path_ours, attr_list, attr_strength_our, seq)

        # combined_image_stack = np.vstack([stack_ifg, stack_gs, stack_sf, stack_our])
        combined_image_stack = stack_gs

        img_save_name = str(id) + '_compare_mat_' + str(attr_list) + '_seq_' + str(seq) + '.jpg'
        img_save_path = os.path.join(dst_path, img_save_name)

        save_img = Image.fromarray(np.array(combined_image_stack, np.uint8)).convert('RGB') 
        print("Saving image stack at: ", img_save_path) 

        save_img.save(img_save_path) 


if __name__ == "__main__":
    compare_main()
