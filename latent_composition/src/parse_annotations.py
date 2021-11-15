import numpy as np
import csv
import pandas as pd
import os
import shutil


# Function to read annotation file line by line and extract relevent annotations 
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


def filter_annotations(anno_file_path, att_set, data_size):
    atts_df = read_annotations(anno_file_path, data_size)

    att = att_set[1]

    atts_filtered = atts_df[att_set]

    atts_filtered_pos = atts_filtered[atts_filtered[att] == '1'].sample(500)
    atts_filtered_neg = atts_filtered[atts_filtered[att] == '-1'].sample(500)

    print("Positives shape: {}".format(atts_filtered_pos.shape))
    print("Negatives shape: {}".format(atts_filtered_neg.shape))

    print("pos head: ", atts_filtered_pos.head())
    print("neg head: ", atts_filtered_neg.head()) 

    return atts_filtered_pos, atts_filtered_neg


# This function will copy the dataset images for the indices present in data index file from src path to dst path
def filter_image_data(data_index_file, src_img_path, src_mask_path, dst_img_path, dst_mask_path):
    data_index = pd.read_csv(data_index_file)
    
    print("{} data index shape: {}".format(data_index_file, data_index.shape))

    for i in range(0, data_index.shape[0]):
        fn = data_index.iloc[i]['file_name']

        fp_si = os.path.join(src_img_path, fn)
        fp_sm = os.path.join(src_mask_path, fn)
        # print("fp_si: {}, fp_sm: {}".format(fp_si, fp_sm))

        # Note: Currently only copyieng the images, not the segmasks as the directory structure for segmentation mask is bit random 
        shutil.copy(fp_si, dst_img_path)  
        # shutil.copy(fp_sm, dst_mask_path)

    print("copied {} image files".format(data_index.shape[0]))     


# Remove all the files from the given input folder
def clean_folder(root_dir):
    for file_ in os.listdir(root_dir):
        os.remove(os.path.join(root_dir, file_))


def main():
    # Reading the annotation files and saving the annotation for desired attribute Eyeglasses, Wearing_Hat
    
    """
    anno_file_path = '../CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'
    data_size = 30*1000
    atts_filtered_pos, atts_filtered_neg = filter_annotations(anno_file_path, ['file_name','Wearing_Hat'], data_size)

    atts_filtered_pos.to_csv('hat_pos_500.csv')
    atts_filtered_neg.to_csv('hat_neg_500.csv') 
    """

    # Using the above filtered data to copy the corresponding images and segmentation masks to a separate folder
    root_path = '../CelebAMask-HQ/' 
    src_img_path = os.path.join(root_path, 'CelebA-HQ-img')
    src_mask_path = os.path.join(root_path, 'CelebA-HQ-mask-anno')


    dst_img_path = os.path.join(root_path, 'data_filtered/img')
    dst_mask_path = os.path.join(root_path, 'data_filtered/anno-mask')

    clean_folder(dst_img_path)
    clean_folder(dst_mask_path)

    data_index_files = ['smile_pos_500.csv', 'smile_neg_500.csv', 'hat_pos_500.csv', 'hat_neg_500.csv', 'eyeglasses_pos_500.csv', 'eyeglasses_neg_500.csv']
    for dif in data_index_files:
        filter_image_data(dif, src_img_path, src_mask_path, dst_img_path, dst_mask_path)

if __name__ == "__main__":
    main()


