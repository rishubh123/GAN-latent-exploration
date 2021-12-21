"""
Author: Rishubh Parihar
Date: 15/11/2011
Description: In this module we there are functions to create dataset with required image pairs. Essentially, we will create
csv files for having original and transformed image filenames. These filenames can be manually added by creating augmentations 
and filtering out good augmentations or we can create extract filenames directly from the csvs corresponding to the positive 
and negative images for any attribute. 

"""


import numpy as np
import pandas as pd 
import os

# This function will generate a csv by creating pairs of images with (augmented) and without attribute from one folder having all these image.
def generate_csv_for_images(img_folder, dst_dir):
    # Here we will create a csv for each single image and all of its possible augmentations
    img_list_hat = ['533.jpg', '1547.jpg', '18277.jpg', '21765.jpg']
    img_list_eye_g = ['27995.jpg', '8167.jpg']

    # Creating files for augmented images with hat 
    for img_nm in img_list_hat:
        map_mat = []
        df_save_path = os.path.join(dst_dir, img_nm[:-4] + '_hat_transforms.csv') 
        
        print("Df save name: ", df_save_path) 
        for file in os.listdir(img_folder):
            # print(" condition: ", file, img_nm)
            if (file[:3] == img_nm[:3] and file[-7:] == 'hat.jpg'):
                row = [img_nm, file]
                map_mat.append(row)
            
        map_mat = np.array(map_mat)
        map_df = pd.DataFrame(data = map_mat, columns = ['Original_img', 'Transformed_img']) 

        print("save mat: ", map_df) 
        map_df.to_csv(df_save_path)
        print("Saving dataframe ... ", df_save_path)  


    # Creating files for augmented images with eyeg
    for img_nm in img_list_eye_g:
        map_mat = []
        df_save_path = os.path.join(dst_dir, img_nm[:-4] + '_eye_g_transforms.csv')
        for file in os.listdir(img_folder):
            if (file[:3] == img_nm[:3] and file[-9:] == 'eye_g.jpg'):
                row = [img_nm, file]
                map_mat.append(row)

        map_mat = np.array(map_mat)
        map_df = pd.DataFrame(data = map_mat, columns = ['Original_img', 'Transformed_img'])

        print("save mat: ", map_df)  
        map_df.to_csv(df_save_path)
        print("Saving dataframe ... ", df_save_path)  


# This function will create a csv with the original file_name and its transformed version for attribute interpolation
def create_csv_for_transformations(src_folder_path, save_file_name, att_suffix): 
    fnames = [fn for fn in os.listdir(src_folder_path)]

    data_table = []
    # Traversing the originals
    for fn in fnames:
        if (fn[-5:] != att_suffix):
            prefix_name = fn[:-4]
            str_len = len(prefix_name)
            tforms_names = []
            for tn in fnames:
                if (tn[:str_len] == prefix_name and tn[-5:] == att_suffix):
                    data_table.append([fn, tn])

    
    data_table = np.array(data_table)
    print("generated data table: ", data_table)
    df = pd.DataFrame(columns=['Original_img', 'Transformed_img'], data = data_table)

    print("Saving the csv to location: ", save_file_name) 
    df.to_csv(save_file_name) 




if __name__ == "__main__":
    # 1.3
    # Once we have copied the images and the semgnetation masks into a separate folder we can then create pairs of images for further processing 
    # Generating csv for each of the image files and its various augmentations for hat and eyeglasses transformations 
    
    # This folder contains all the augmentations which are manually filtered for latent discovery 
    # aug_img_folder = '../CelebAMask-HQ/data_filtered/filter_augmentations_id/'   
    # dst_dir = '../data_files/'
    # generate_csv_for_images(aug_img_folder, dst_dir)    

    # 2.2
    att = 'bald' 
    att_suffix = 'd.jpg'
    src_folder_path = '../CelebAMask-HQ/data_filtered/renew/augmentations/filtered_att_dirs_dataset/' + att 
    save_name = '../data_files/att_dirs_fs/att_dirs_fs_' + att + '.csv'
    create_csv_for_transformations(src_folder_path, save_name, att_suffix)  
    
