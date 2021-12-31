"""
Author: Rishubh Parihar
Date: 15/11/2011

Description: This file has all the utility functionality to read the CELEBA-HQ-MASK dataset. Here we curate dataset for GAN latent 
sapce exploration project. It first filters out a given number of positive and negative examples given the attribute type (eg. eyeglass, smile, wearining_hat)
After filtering out these example images it will create separate csv files for a single attribute and single label(positive/negative)
There is addional functions implemented to copy only the set of images present in the csv files into separate folders for 
fast and easy processing. Also, part segmentation mask can also be copied into a separate folder from the image names from csv. 
"""   



import numpy as np
import csv
import pandas as pd
import os
import shutil

global debug_flag 

# Utility function to read annotation file line by line and extract relevent annotations. This will create a final pandas dataframe for the annotation dataset
def read_annotations(anno_file_path, data_size):
    anno_file = open(anno_file_path, 'r') 

    n = anno_file.readline()
    attributes = ['file_name'] + anno_file.readline()[:-1].split(' ') 

    print("attributes: ", attributes)
    anno_matrix = []

    for i in range(0, data_size): 
        raw_line = anno_file.readline()[:-1].split(' ')
        
        pro_line = [raw_line[0]] + raw_line[2:] # Removing one space entry 

        anno_matrix.append(pro_line)
        # print("{} line: {}".format(i, pro_line))

    anno_matrix = np.array(anno_matrix)
    atts_df = pd.DataFrame(data = anno_matrix, columns = attributes) 

    # print("final df: {}".format(atts_df))

    return atts_df


# Utility function which will read the annotation file and will filter out two lists one having the positive and negative label for the given attribute 
def filter_annotations(anno_file_path, df_columns, data_size):
    atts_df = read_annotations(anno_file_path, data_size)
    df_columns.append('Male')

    att = df_columns[1] #[file_name, att]
    
    atts_filtered = atts_df[df_columns]

    # We are sampling and saving only the sampled 100 images 
    atts_filtered_pos = atts_filtered[atts_filtered[att] == '1'].sample(100)
    atts_filtered_neg = atts_filtered[atts_filtered[att] == '-1'].sample(100)

    if (debug_flag):
        print("Filtering examples for attribute ", att)
        print("Positives shape: {}".format(atts_filtered_pos.shape))
        print("Negatives shape: {}".format(atts_filtered_neg.shape))

        print("pos head: ", atts_filtered_pos.head())
        print("neg head: ", atts_filtered_neg.head()) 

    return atts_filtered_pos, atts_filtered_neg


# This function will copy the dataset images present in the data_index_files (att's pos and neg image csvs), from the original dataset into a destination folder 
def filter_image_data(data_index_file, src_img_path, dst_img_path):   
    data_index = pd.read_csv(data_index_file)
    
    if (debug_flag):
        print("{} data index shape: {}".format(data_index_file, data_index.shape))

    for i in range(0, data_index.shape[0]):
        fn = data_index.iloc[i]['file_name']
        fp_si = os.path.join(src_img_path, fn)

        # Note: only copying the images not the segmasks as the directory structure for segmentation mask is different
        shutil.copy(fp_si, dst_img_path)  

    if (debug_flag):
        print("copied {} image files".format(data_index.shape[0]))      



# This function will copy the segmentation masks for the image names present in data_index_files (att's pos and neg images), from the original dataset into a destination folder.
def filter_mask_data(data_index_file, segmask_list, src_mask_path, dst_mask_path):
    data_index = pd.read_csv(data_index_file)
    
    if (debug_flag):
        print("{} data index shape: {}".format(data_index_file, data_index.shape))

    idx = 0
    for i in range(0, data_index.shape[0]):
        fp_n = data_index.iloc[i]['file_name']
        fp_n = str(format(int(fp_n[:-4]), '05d')) 
        fp = os.path.join(src_mask_path, fp_n) 

        fp_s_list = [fp + '_' + att_mask + '.png' for att_mask in segmask_list]    

        
        for fp_s in fp_s_list:
            if (os.path.exists(fp_s)):
                shutil.copy(fp_s, dst_mask_path) 
                idx+=1
        

    if (debug_flag):  
        print("copied {} masks files".format(idx)) 


# This function will extract annotations for all the attributes and combines them into a single folder 
def unify_annotation_masks(image_anno_root):
    folders = [os.path.join(image_anno_root, str(i)) for i in range(0,15)]
    dst_combined_path = os.path.join(image_anno_root, 'combined_annos_full/')

    # for folder in os.listdir(image_anno_root):
    #     print("folder: ", folder)

    for folder in folders:
        print("Processing {} folder".format(folder))
        for img_nm in os.listdir(folder):
            img_path = os.path.join(folder, img_nm)
            shutil.copy(img_path, dst_combined_path)   


# Remove all the files from the given input folder
def clean_folder(root_dir):
    for file_ in os.listdir(root_dir):
        os.remove(os.path.join(root_dir, file_))


# Creating a test set with 500 images
def create_test_set(train_set_path, images_path, dst_images_path):  
    copied_images = 0 
    train_set = pd.read_csv(train_set_path)
    print(train_set.head())

    # Creating a list of all the original image names 
    train_img_list = list(train_set['img_name'])
    print("train image list: ", len(train_img_list))

    for img in os.listdir(images_path):
        if (not img in train_img_list):
            fp = os.path.join(images_path, img)
            shutil.copy(fp, dst_images_path)
            copied_images += 1
        if (copied_images > 500):
            break

    print("Copied {} images.", copied_images)


# This function will process the combined annotation file for CELEBA-HQ-MASK dataset and for each attribute create two csvs one for positives and one for negatives
# inputs:
#    anno_file_path - The file path for the annotation file having all the attributes mentioned
#    dst_file_path - The folder path where the generated csv for the attribute will be saved after processing
#    att - The attribute for which the dataset is to be filtered: eg. smile, eyeglasses, wearing_hat
#    data_size - The overall data set size which will be processed to create the filter datas size 
# functionality: This function will create two lists and will save them at the destination file path location
def create_attribute_datasets(anno_file_path, dst_folder_path, att, data_size):
    atts_filtered_pos, atts_filtered_neg = filter_annotations(anno_file_path, ['file_name',att], data_size)

    att_pos_fn = os.path.join(dst_folder_path, att + '_pos.csv')
    att_neg_fn = os.path.join(dst_folder_path, att + '_neg.csv')

    if (debug_flag):
        print("att pos fn: ", att_pos_fn)
        print("att neg fn: ", att_neg_fn)

    atts_filtered_pos.to_csv(att_pos_fn)
    atts_filtered_neg.to_csv(att_neg_fn) 


def main():
    # Annotation file path containing annotations for all the attributes present in CELEBA-HQ-MASK dataset
    anno_file_path = '../CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'
    dst_folder_path = '../data_files/attribute_datasets/'
    # Total data size to be processed for filtering 
    data_size = 30*1000
    # att_list = ['Smiling', 'Mouth_Slightly_Open', 'No_Beard', 'Mustache', 'Narrow_Eyes', 'Wearing_Hat', 'Black_Hair', 'Blond_Hair',
    #             'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair', 'Receding_Hairline']      
    att_list = ['Bald']    

    # 1.1 Reading the annotation files and saving the annotation for desired attribute Eyeglasses, Wearing_Hat 
    """
    # Traversing over the attribute list and calling create_attribute_datasets for each of the attribute 
    for att in att_list: 
        create_attribute_datasets(anno_file_path, dst_folder_path, att, data_size)    
    """

    # 1.2 
    # Using the above filtered data to copy the corresponding images and segmentation masks to a separate folder
    """
    root_path = '../CelebAMask-HQ/'  
    src_img_path = os.path.join(root_path, 'CelebA-HQ-img') 
    src_mask_path = os.path.join(root_path, 'CelebAMask-HQ-mask-anno/combined_annos_full') 

    # File and folder location to save the images and segmenation masks for all the images in the correspoding positive and negative lists for all the attributes. 
    dst_img_path = os.path.join(root_path, 'data_filtered/renew/img')  
    dst_mask_path = os.path.join(root_path, 'data_filtered/renew/anno-mask')
    data_files_path_att = '../data_files/attribute_datasets' 

    data_index_names_pos = [att + '_pos.csv' for att in att_list]
    data_index_names_neg = [att + '_neg.csv' for att in att_list]
    data_index_names = data_index_names_pos + data_index_names_neg
    data_index_files = [os.path.join(data_files_path_att, din) for din in data_index_names] 

    # Copying all the masks for all the attributes in a single folder. Only to run once. 
    # masks_root_dir = '../CelebAMask-HQ/CelebAMask-HQ-mask-anno/'
    # unify_annotation_masks(masks_root_dir)  

    
    if (debug_flag):
        print("data index files for copyieng image and mask data: ", data_index_files)
    
    segmask_list = ['hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_lip', 'mouth', 'u_lip', 'neck', 'nose', 'skin', 'cloth', 'l_ear', 'r_ear',
                    'hat', 'eye_g']

    for dif in data_index_files:
        # Copying the image files into a separate folder
        filter_image_data(dif, src_img_path, dst_img_path)         
        # Copying the segmentation masks into a separate folder 
        filter_mask_data(dif, segmask_list, src_mask_path, dst_mask_path) 
    """

    # 1.3 Create test set by selecting images which are not in the training set for attribute estimation
    train_set_path = '../data_files/att_dirs_dataset_fs/att_dirs_fs_combined.csv'
    images_path = '../CelebAMask-HQ/CelebA-HQ-img/'
    dst_images_path = '../CelebAMask-HQ/data_filtered/test500'
    # Creating a folder for saving the test images 
    if (not os.path.exists(dst_images_path)):
        os.mkdir(dst_images_path)
    create_test_set(train_set_path, images_path, dst_images_path)


if __name__ == "__main__":
    debug_flag = True
    main() 


