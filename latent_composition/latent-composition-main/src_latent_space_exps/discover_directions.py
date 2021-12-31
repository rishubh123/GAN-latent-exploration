"""
Date: 29/12/2021
This function has all the implementation for estimating the linear latent direction for any attribute in question. 
- Here we will take a csv of positive and negative pairs of images and use their already saved latent codes in W+ space
- Then we expriment with multiple variations to estimate the latent directions for any given attribute using these pair wise latent codes
- This is one of the main implmentation block for our contribution. 
- wo_id: without identity, the pairs are selected randomly and do not have same identity 
- id: identity, the pairs are generated by augmentation and have same identity
"""

import os
import pandas as pd
import numpy as np
import pickle 


"""
# Deprecated, not in use
# This function takes as input a single input image and all its transformations for any given attribute and then estimate the directions corresponding
# to the arribute by averaging over all the original and transformed image pairs 
# Computing the directions for the id pairs of a image and all its transformation 
def compute_separate_directions_id_pairs():
    atts_list = ['eyeglasses', 'hat']
    data_files_root = '../data_files'
    embds_path_root = '../CelebAMask-HQ/data_filtered/latents'

    separate_data_files = ['533_hat_transforms.csv',
                           '1547_hat_transforms.csv',
                           '8167_eye_g_transforms.csv', 
                           # '18277_hat_transforms.csv',
                           '21765_hat_transforms.csv',
                           '27995_eye_g_transforms.csv']  
    
    separate_data_files = [os.path.join(data_files_root, sdf) for sdf in separate_data_files] 


    separate_data_files = ['synth_aug_55_data_eyeglasses.csv', 'synth_aug_55_data_hat.csv'] 
    separate_data_files = [os.path.join(data_files_root, sd) for sd in separate_data_files]

    for data_file in separate_data_files:
        compute_separate_single_direction_pairwise(data_file, data_files_root, embds_path_root) 
"""

"""
# Deprecated, not in use | all the functionality is moved into the new functions 
# 1-n config. This function computes pairwise direction for a single source image and its multiple transformations 
def compute_separate_single_direction_pairwise(pairwise_file_path, data_files_root, embds_path_root):
    print("pair-wise file path: ", pairwise_file_path)
    files_ = pd.read_csv(pairwise_file_path)
    print("files head of data:", files_.head())
    latent_diffs = []
    num_pairs = min(5, files_.shape[0])
    latent_dirs_dict = {}
    src_img_name = ''

    for i in range(num_pairs):
        img_orig_name = files_.iloc[i]['Original_Image']
        img_augname = files_.iloc[i]['Augmented_Image']  
        src_img_name = img_orig_name[:-4] # Img orig is same in all the rows  

        orig_embd_path = os.path.join(embds_path_root, img_orig_name[:-4] + '.npy')
        aug_embd_path = os.path.join(embds_path_root, img_augname[:-4] + '.npy')

        orig_latent = np.load(orig_embd_path)
        aug_latent = np.load(aug_embd_path)
        diff_latent = aug_latent - orig_latent 

        # Saving the Computed latent directions into a dictionary with the id as the augmented image name 
        latent_dirs_dict[img_augname] = np.array(diff_latent)

    print("latent diffs dict shape: {}".format(len(latent_dirs_dict.keys())))

    fn = 'all_latent_dirs_pairwise' + str(num_pairs) + '_' + src_img_name + '.pkl' 
    latent_save_name = os.path.join(data_files_root, fn)

    print("saving latent for: ", pairwise_file_path, " at: ", latent_save_name)   
    
    # Store data (serialize)
    with open(latent_save_name, 'wb') as handle:
        pickle.dump(latent_dirs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Load data (deserialize)
    with open(latent_save_name, 'rb') as handle:
        unserialized_data = pickle.load(handle) 

    # print("Checking the saved and read pkl files: ", (latent_dirs_dict == unserialized_data).any())
"""


# This function takes all the n*n pairs from the positive and negative list and and estimate the directions, finally the directions are averaged and saved into a numpy array 
# This function takes the average of all the direction before saving them into an numpy array 
def compute_direction_wo_id_att(att, data_files_root, embds_path_root, n):
    pos_file_path = os.path.join(data_files_root, att + '_pos_500.csv')
    neg_file_path = os.path.join(data_files_root, att + '_neg_500.csv')

    print("pos file path: ", pos_file_path)
    print("neg file path: ", neg_file_path)
    pos_files = pd.read_csv(pos_file_path)
    neg_files = pd.read_csv(neg_file_path)

    # n = len(pos_files['file_name'])
    print("iterating over {} files to estimate the direction".format(n)) 

    # Looping over i and j to get all the pair-wise differences between the latent code 
    latent_diffs = []
    for i in range(0, n):  
        for j in range(0, n):
            # Taking ith image and all the pairs for j belong to (0,n)
            img_pos_name = pos_files.iloc[i]['file_name']
            img_neg_name = neg_files.iloc[j]['file_name']
            # print("computing direction for: {}, {}".format(i,j))

            pos_embd_path = os.path.join(embds_path_root, img_pos_name[:-4] + '.npy')
            neg_embd_path = os.path.join(embds_path_root, img_neg_name[:-4] + '.npy')

            pos_latent = np.load(pos_embd_path)
            neg_latent = np.load(neg_embd_path)
            diff_latent = pos_latent - neg_latent

            latent_diffs.append(diff_latent)

    latent_diffs = np.array(latent_diffs)
    latent_diffs_avg = np.mean(latent_diffs, axis=0) 
    print("latent diffs shape: {}, latent diffs avg shape: {}".format(latent_diffs.shape, latent_diffs_avg.shape))

    fn = 'avg_latent_dir_nc2_' + str(n) + '_' + att + '.npy' 
    latent_save_name = os.path.join(data_files_root, fn)

    print("saving latent for: ", att, " at: ", latent_save_name)    
    np.save(latent_save_name, latent_diffs_avg)


# Computing directions for every pair formed by n-positive and n-negative images of the atrributes given to the network by non-identity pairs
def compute_directions_wo_id():    
    atts_list = ['eyeglasses', 'hat', 'smile']
    data_files_root = '../data_files'
    embds_path_root = '../CelebAMask-HQ/data_filtered/latents'

    n = 50 # Number of samples for each pos and neg categories 
    # Iterating over all the attributes and finding the independent directions for each attribute 
    for i in range(0, len(atts_list)):
        compute_direction_wo_id_att(atts_list[i], data_files_root, embds_path_root, n)


# Computing the directions for the pairs given in a csv file, these pairs are augmented positive and negative images for any given attribute 
# This function saves all the directions in a file without averaging or normalizing 
def compute_direction_id_att(att, data_files_root, pairwise_file_path, embds_path_root):
    print("pair-wise file path: ", pairwise_file_path)
    files_df = pd.read_csv(pairwise_file_path) 

    latent_diffs = []
    n_pairs = files_df.shape[0]
    print("Taking difference for {} pairs ... ".format(n_pairs))
    # exit()  

    # Iterating over all the pairs from the dataframe 
    for pair_id in range(n_pairs):
        img_orig_name = files_df.iloc[pair_id]['Original_img']
        img_augname = files_df.iloc[pair_id]['Transformed_img']

        orig_embd_path = os.path.join(embds_path_root, img_orig_name[:-4] + '.npy')
        aug_embd_path = os.path.join(embds_path_root, img_augname[:-4] + '.npy')

        orig_latent = np.load(orig_embd_path)
        aug_latent = np.load(aug_embd_path)
        diff_latent = aug_latent - orig_latent

        latent_diffs.append(diff_latent)

    latent_diffs = np.array(latent_diffs) 
    # print("latent difffs shape: {}".format(latent_diffs.shape))

    fn = 'latent_db_dir_id_' + str(n_pairs) + '_' + att + '.npy'
    latent_save_name = os.path.join(data_files_root, fn)

    # Saving the difference of latent directions at the file location 
    print("saving latent for: ", att, " at: ", latent_save_name)
    np.save(latent_save_name, latent_diffs)

# This function finds the directions for all the augmented pairs created for each of the given attribute.  
def compute_direction_id():
    embds_path_root = '../../CelebAMask-HQ/data_filtered/renew/latents'
    # att_list = ['bald', 'bang', 'eye_g', 'hat', 'smile']
    att_list = ['pose']
    data_files_root = '../../data_files/estimated_dirs/'
    pairwise_path_root = '../../data_files/att_dirs_dataset_fs/'
    pairwise_paths = [os.path.join(pairwise_path_root, 'att_dirs_fs_' + att) + '.csv' for att in att_list]

    # Iterating over all the attributes for processing them independently 
    for idx in range(0, 1):
        # Selecting the idx^th attribute and corresponding csv file path 
        compute_direction_id_att(att_list[idx], data_files_root, pairwise_paths[idx], embds_path_root)    


if __name__ == "__main__":  
  print("running main ...")   
  print("computing all directions ...")    
  
  # Computing the directions for all the pairs formed by creating without identity pairs 
  # compute_directions_wo_id() 

  # Computing the directions for the augmented pairs of positive and negative images 
  compute_direction_id()