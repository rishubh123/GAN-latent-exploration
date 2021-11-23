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



# This function will copy the dataset masks for the indices present in data index file from src path to dst path
def filter_mask_data(data_index_file, src_mask_path, dst_mask_path):
    data_index = pd.read_csv(data_index_file)
    
    print("{} data index shape: {}".format(data_index_file, data_index.shape))

    idx = 0
    for i in range(0, data_index.shape[0]):
        fp_n = data_index.iloc[i]['file_name']
        fp_n = str(format(int(fp_n[:-4]), '05d')) 
        fp = os.path.join(src_mask_path, fp_n)

        fp_si_eyeg = fp + '_eye_g.png'
        fp_si_hat = fp + '_hat.png' 

        if (os.path.exists(fp_si_eyeg)):
            # print(fp_si_eyeg)
            shutil.copy(fp_si_eyeg, dst_mask_path)
            idx+=1
            # exit()

        if (os.path.exists(fp_si_hat)):
            # print(fp_si_hat)
            shutil.copy(fp_si_hat, dst_mask_path)
            idx+=1

    print("copied {} masks files".format(idx))      


# This function will extract annotations for few required attributes and use them to copy into a single folder
def unify_annotation_masks(image_anno_root):
    folders = [os.path.join(image_anno_root, str(i)) for i in range(0,15)]
    dst_combined_path = os.path.join(image_anno_root, 'combined_annos/')

    # for folder in os.listdir(image_anno_root):
    #     print("folder: ", folder)

    for folder in folders:
        print("Processing {} folder".format(folder))
        for img_nm in os.listdir(folder):
            img_path = os.path.join(folder, img_nm)

            if (img_nm[-9:] == 'eye_g.png' or img_nm[-7:] == 'hat.png'):
                shutil.copy(img_path, dst_combined_path) 


# Remove all the files from the given input folder
def clean_folder(root_dir):
    for file_ in os.listdir(root_dir):
        os.remove(os.path.join(root_dir, file_))

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



def main():
    # Reading the annotation files and saving the annotation for desired attribute Eyeglasses, Wearing_Hat
    
    """
    anno_file_path = '../CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'
    data_size = 30*1000
    atts_filtered_pos, atts_filtered_neg = filter_annotations(anno_file_path, ['file_name','Wearing_Hat'], data_size)

    atts_filtered_pos.to_csv('hat_pos_500.csv')
    atts_filtered_neg.to_csv('hat_neg_500.csv') 
    """

    """ 
    # Using the above filtered data to copy the corresponding images and segmentation masks to a separate folder
    root_path = '../CelebAMask-HQ/' 
    src_img_path = os.path.join(root_path, 'CelebA-HQ-img') 
    src_mask_path = os.path.join(root_path, 'CelebAMask-HQ-mask-anno/combined_annos') 


    dst_img_path = os.path.join(root_path, 'data_filtered/img') 
    dst_mask_path = os.path.join(root_path, 'data_filtered/anno-mask')
    data_files_path = '../data_files'

    # Cleaning files to remove the older files 
    # clean_folder(dst_img_path)
    # clean_folder(dst_mask_path)

    data_index_names = ['smile_pos_500.csv', 'smile_neg_500.csv', 'hat_pos_500.csv', 'hat_neg_500.csv', 'eyeglasses_pos_500.csv', 'eyeglasses_neg_500.csv']
    data_index_files = [os.path.join(data_files_path, din) for din in data_index_names] 

    for dif in data_index_files:
        # filter_image_data(dif, src_img_path, src_mask_path, dst_img_path, dst_mask_path)
        filter_mask_data(dif, src_mask_path, dst_mask_path) 

    # Copying all the masks for eyeglass and hat attribute into a combined filter 
    # unify_annotation_masks(src_mask_path)
    """

    # Generating csv for each of the image files and its various augmentations for hat and eyeglasses transformations 
    img_folder = '../CelebAMask-HQ/data_filtered/augmented_id_filtered/'  
    dst_dir = '../data_files/'
    generate_csv_for_images(img_folder, dst_dir)

if __name__ == "__main__":
    main() 


