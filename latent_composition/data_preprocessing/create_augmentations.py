"""
Author: Rishubh Parihar
Date: 15/11/2011

Description: This module builds the functionality of creating augmented images using the part-wise segmentation mask.
Given any attribute in question, it first extracts the set of images marked positive and negative for that attribute and
then create an augmentation by pasting the attribute from the positive image to the negative image using the partwise 
segmentation masks. The dataset is eye-aligned so directly pasting results in good augmentations for some images. However, 
for some cases the augmentation might be very wrong. Currently we create multiple of such augmenations by direct pasting
and select out some of them manually for further processing.
Improvements:
1. Using landmarkd points for better alignment of the positive and negative image
2. Creating image warps from the negative image to the positive image to get perfect alignment before copying the 
   region of interests 
"""

import numpy as np
import cv2
import pandas as pd
import math  
import os 
import shutil 

global debug_flag

# This utility function will take and image and mark the landmark points on it along with the order in which they occur on the annotataions
def display_landmarks(img, landmarks): 
    if (debug_flag):
        print("landmarks shape: ", landmarks.shape)  
    img_disp = img.copy() 

    for i in range(0, 68):
        x_id = i*2 
        y_id = i*2 + 1

        x = int(landmarks[x_id])
        y = int(landmarks[y_id])
        # print("x, y: ", x, y)

        cv2.putText(img_disp, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite('landmark_marked.png', img_disp)


    
# ----- Deprecated -----
# This function creates the augmentation by first aligning using the landmarks points of the negative and the positive attribute image. 
# Then using the segmentation mask to create augmented versions of the original image. 
# The complete functionality is not implemented only, eye alignment logic is present her. Can be improved by incorporating the 
# the alignment of the facial part for which we are working for creating augmentations 
def create_augmentation_with_alignment(data_root_dir, src_img_name, att_img_name, att, landmarks_df):
    src_img_path = os.path.join(data_root_dir, 'img', src_img_name) # Src image on which we will augment different attributes
    att_img_path = os.path.join(data_root_dir, 'img', att_img_name) # Attribute image from which we will extract visual attributes like eye_glasses or hat

    mask_path_root = str(format(int(att_img_name[:-4]), '05d')) # All the masks are saved in the format of 000xx.png
    mask_path = os.path.join(data_root_dir, 'anno-mask', mask_path_root + att + '.png') # Mask for the attribute image 

    if (not os.path.exists(mask_path)):
        print("mask path: ", mask_path)
        print("Mask path does not exist, returning ... ") 
        return 

    # Image names for saving 
    image_save_name = os.path.join(data_root_dir, 'augmented_id_multiple', src_img_name[:-4] + '_' + att_img_name[:-4] + '_' + att + '.jpg')
    src_img_save_path = os.path.join(data_root_dir, 'augmented_id_multiple', src_img_name) 

    # print("src img name: ", src_img_name)
    # print("att img name: ", att_img_name)
    # print("mask path: ", mask_path)

    src_img = cv2.imread(src_img_path)
    att_img = cv2.imread(att_img_path)
    mask_img_orig = cv2.imread(mask_path) # ORiginal image of 512x512
    mask_img = cv2.resize(mask_img_orig, (src_img.shape[0], src_img.shape[1])) # Resizing mask image to match the src image to 1024x1024

    # Saving the source image into the save folder for easy processing 
    cv2.imwrite(src_img_save_path, src_img)

    src_img_ldmks = landmarks_df.loc[landmarks_df['file_name'] == src_img_name]
    att_img_ldmks = landmarks_df.loc[landmarks_df['file_name'] == att_img_name]

    if (src_img_ldmks.shape[0] == 0):
        return 

    if (att_img_ldmks.shape[0] == 0):
        return 

    src_img_ldmks = src_img_ldmks.values[0,2:]
    att_img_ldmks = att_img_ldmks.values[0,2:]


    # display_landmarks(src_img, src_img_ldmks)

    # print("src img_ldmks:", len(src_img_ldmks), type(src_img_ldmks))
    # print("att img_ldmks:", len(att_img_ldmks), type(att_img_ldmks))

    # central location of the right eye of source image | 42-47
    eye_r_x_src = (src_img_ldmks[42*2] + src_img_ldmks[43*2] + src_img_ldmks[44*2] + src_img_ldmks[45*2] + src_img_ldmks[46*2] + src_img_ldmks[47*2]) / 6 
    eye_r_y_src = (src_img_ldmks[42*2 + 1] + src_img_ldmks[43*2 + 1] + src_img_ldmks[44*2 + 1] + src_img_ldmks[45*2 + 1] + src_img_ldmks[46*2 + 1] + src_img_ldmks[47*2 + 1]) / 6 

    # central location of the left eye of source image | 36-41
    eye_l_x_src = (src_img_ldmks[36*2] + src_img_ldmks[37*2] + src_img_ldmks[38*2] + src_img_ldmks[39*2] + src_img_ldmks[40*2] + src_img_ldmks[41*2]) / 6 
    eye_l_y_src = (src_img_ldmks[36*2 + 1] + src_img_ldmks[37*2 + 1] + src_img_ldmks[38*2 + 1] + src_img_ldmks[39*2 + 1] + src_img_ldmks[40*2 + 1] + src_img_ldmks[41*2 + 1]) / 6 

    # central location of the right eye of the attribute image | 42-47
    eye_r_x_att = (att_img_ldmks[42*2] + att_img_ldmks[43*2] + att_img_ldmks[44*2] + att_img_ldmks[45*2] + att_img_ldmks[46*2] + att_img_ldmks[47*2]) / 6 
    eye_r_y_att = (att_img_ldmks[42*2 + 1] + att_img_ldmks[43*2 + 1] + att_img_ldmks[44*2 + 1] + att_img_ldmks[45*2 + 1] + att_img_ldmks[46*2 + 1] + att_img_ldmks[47*2 + 1]) / 6 

    # central location of the left eye of the attribute image | 36-41
    eye_l_x_att = (att_img_ldmks[36*2] + att_img_ldmks[37*2] + att_img_ldmks[38*2] + att_img_ldmks[39*2] + att_img_ldmks[40*2] + att_img_ldmks[41*2]) / 6 
    eye_l_y_att = (att_img_ldmks[36*2 + 1] + att_img_ldmks[37*2 + 1] + att_img_ldmks[38*2 + 1] + att_img_ldmks[39*2 + 1] + att_img_ldmks[40*2 + 1] + att_img_ldmks[41*2 + 1]) / 6 

    # Computing the shift in the eye locations in the two images 
    shift_x = int(eye_l_x_src - eye_l_x_att)
    shift_y = int(eye_l_y_src - eye_l_y_att)

    # print("shift x: ", shift_x, " shift y: ", shift_y)

    mask_shifted = np.zeros(mask_img.shape)
    att_img_shifted = np.zeros(att_img.shape)

    l,w = mask_img.shape[0], mask_img.shape[1]
    mask_shifted[max(shift_y,0): min(l, l+shift_y), max(shift_x,0): min(w, w+shift_x)] = mask_img[max(-shift_y, 0): min(l, l-shift_y), max(-shift_x,0): min(w, w-shift_x)]
    att_img_shifted[max(shift_y,0): min(l, l+shift_y), max(shift_x,0): min(w, w+shift_x), :] = att_img[max(-shift_y, 0): min(l, l-shift_y), max(-shift_x,0): min(w, w-shift_x), :]

    # Computing the distance between the eyes and the angle between two eyes for source image 
    src_eye_dist = math.sqrt((eye_r_x_src - eye_l_x_src)*(eye_r_x_src - eye_l_x_src) + (eye_r_y_src - eye_l_y_src)*(eye_r_y_src - eye_l_y_src)) 
    src_eye_angle = (eye_r_y_src - eye_l_y_src) / (eye_r_x_src - eye_l_x_src)   
    # print("src img eye dist:", src_eye_dist, " src eye angle: ", src_eye_angle)

    # Computing the distance between the wyes and angle between the two eyes for attribute image 
    att_eye_dist = math.sqrt((eye_r_x_att - eye_l_x_att)*(eye_r_x_att - eye_l_x_att) + (eye_r_y_att - eye_l_y_att)*(eye_r_y_att - eye_l_y_att))
    att_eye_angle = (eye_r_y_att - eye_l_y_att) / (eye_r_x_att - eye_l_x_att)
    # print("att img eye dist:", att_eye_dist, " att eye angle: ", att_eye_angle)

    size_ratio = src_eye_dist / att_eye_dist 
    # mask_img_resized = cv2.imresize(mask_img, (size_ratio * ))

    mask_shifted = mask_shifted/255.0
    # print("mask shifted range: ", mask_shifted.min(), mask_shifted.max())
    modified_img = mask_shifted * att_img_shifted + (1-mask_shifted) * src_img

    dim = 256
    combined_imgs = np.hstack([cv2.resize(src_img, (dim,dim)), cv2.resize(att_img, (dim, dim)), cv2.resize(mask_img, (dim, dim)), cv2.resize(modified_img, (dim, dim))])
    # cv2.imshow("combined input images", combined_imgs)
    # cv2.waitKey(10000)  

    # Saving temporarily the image stack
    cv2.imwrite('temp/sample_'+src_img_name[:-4] + att_img_name[:-4] + '_' + att + '.png', combined_imgs)
    # Saving only the modified image 
    cv2.imwrite(image_save_name, modified_img) 


# This function creates an augmentation of the negative image given an positive attribute image by using the segmentation mask and fusing these two images.
# This function does not perform additional alignment of facial parts using landmarks points. However, eyes are already aligned in the dataset resuling in
# few good augmentations wo having alignment step. 
def create_augmentations_wo_alignment(data_root_dir, src_img_name, att_img_name, segm, att, change_clr=False):  
    src_img_path = os.path.join(data_root_dir, 'img', src_img_name)  # Source negative image path
    att_img_path = os.path.join(data_root_dir, 'img', att_img_name)  # Source positive image path from which the attribute will be extracted 

    # In which folder the augmentations should be saved 
    folder_path_aug = os.path.join(data_root_dir, 'augmentations', att)
    if (not os.path.exists(folder_path_aug)):
        print("Folder does not exists for augmentation: ", folder_path_aug)
        os.mkdir(folder_path_aug)

    # Image name and path for saving the augmented images, for augmented image, prefix of both the negative and positive images are passed
    aug_image_save_path = os.path.join(folder_path_aug, src_img_name[:-4] + '_' + att_img_name[:-4] + '_' + att + '.jpg')
    src_img_save_path = os.path.join(folder_path_aug, src_img_name) 

    if (debug_flag):
        print("src img name: ", src_img_name)
        print("att img name: ", att_img_name)

    # Reading the images and mask for the given attribute 
    src_img = cv2.imread(src_img_path) 
    att_img = cv2.imread(att_img_path) 
    mask_img = cv2.resize(segm, (src_img.shape[0], src_img.shape[1])) # Resizing mask image to match the src image to 1024x1024

    # Creating the augmentation by mapping the masked region from the negative and unmasked region from the positive attribute image.
    mask_img_norm = mask_img / 255.0
    
    # changing the color of the attribute being copied 
    if (change_clr):
        print("changing color for augmentation")  
        att_img[:,:,1] += 50
        att_img[:,:,0] += 100
        att_img[:,:,2] += 100

    modified_img = mask_img_norm * att_img + (1-mask_img_norm) * src_img       

    # Creating a flip augmentation if we are working with 'pose' attribute
    if (att == 'pose'):
        xc, yc = src_img.shape[0] / 2, src_img.shape[1] / 2
        modified_img = cv2.flip(src_img,1)   
    
    # Dimensions for saving the image 
    dim = 256
    # Creating combined stack of images for better visualization 
    combined_imgs = np.hstack([cv2.resize(src_img, (dim,dim)), cv2.resize(att_img, (dim, dim)), cv2.resize(mask_img, (dim, dim)), cv2.resize(modified_img, (dim, dim))])
    
    if (debug_flag):
        print("Displaying images ...")
        # cv2.imshow("combined input images", combined_imgs)
        # cv2.waitKey(10000)  

    # Saving temporarily the image stack analysis 
    cv2.imwrite('temp/sample_'+src_img_name[:-4] + att_img_name[:-4] + '_' + att + '.png', combined_imgs)
    
    # Saving the source image into the save folder for easy processing and saving the augmented image along with that 
    # print("Saving image to: ", src_img_save_path)
    # print("Saving aug image to: ", aug_image_save_path)
    cv2.imwrite(src_img_save_path, src_img) 
    cv2.imwrite(aug_image_save_path, modified_img) 
    
# This function creates a global segmentation mask by combining all the segmentation mask corresponding to given attribute space
def create_global_segm(data_root_dir, att_img_name, att): 
    mask_path_prefix = str(format(int(att_img_name[:-4]), '05d')) # All the masks are saved in the format of 000xx.png. Initials of the image name 'xxxxx'
    
    # If we are working with any of these below attributes the regions are contained in only one single part segmentation mask
    if (att == 'hat' or att == 'eye_g' or att == 'hair' or att == 'bangs' or att == 'bald'):
        
        mask_reg = att
        # If any of the hair attribute is present we will use the hair mask path
        if (att == 'bangs' or att == 'hair' or att == 'straight_hairs' or att == 'bald'):
            mask_reg = 'hair'

        mask_path = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_' + mask_reg + '.png') # Mask for the attribute image 
        
        if (not os.path.exists(mask_path)):
            print("mask path: ", mask_path)
            print("Mask path does not exist, returning ... ")  
            return False, None

        if (debug_flag):
            print("mask path: ", mask_path)

        mask_img = cv2.imread(mask_path) # ORiginal image of 512x512, we have to resize it  

    # For eye region we have to read two separate semgentation masks, one for the right eye and another one for the left eye 
    elif (att == 'eye'):
        mask_path1 = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_r_eye' + '.png') # Mask path for the right eye
        mask_path2 = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_l_eye' + '.png') # Mask path for the left eye

        if (not (os.path.exists(mask_path1) and os.path.exists(mask_path2))):
            print("mask path: ", mask_path1, " or ", mask_path2)
            print("Mask path does not exist, returning ... ") 
            return False, None 

        mask_img1 = cv2.imread(mask_path1)
        mask_img2 = cv2.imread(mask_path2)

        # combining the two mask images to create a single mask 
        mask_img = mask_img1 + mask_img2

    # For smile and lips processing, we need three part segment masks lower lip, upper lip and mouth for creating a single segm 
    elif(att == 'smile' or att == 'lips'):
        mask_path1 = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_l_lip.png') # Mask path for the right eye
        mask_path2 = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_u_lip.png') # Mask path for the left eye 
        mask_path3 = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_mouth.png') # Mask path for the left eye 

        if (not (os.path.exists(mask_path1) and os.path.exists(mask_path2) and os.path.exists(mask_path3))):
            print("Either of the three mask paths does not exists: ", mask_path1, mask_path2, mask_path3)
            print("Mask path does not exist, returning ... ") 
            return False, None

        mask_img1 = cv2.imread(mask_path1)
        mask_img2 = cv2.imread(mask_path2)
        mask_img3 = cv2.imread(mask_path3)

        # combining the two mask images to create a single mask 
        mask_img = mask_img1 + mask_img2 + mask_img3

    elif(att == 'pose'):
        mask_reg = 'nose'
        mask_path = os.path.join(data_root_dir, 'anno-mask', mask_path_prefix + '_' + mask_reg + '.png')

        mask_img = np.zeros((512,512,3))
        

    return True, mask_img
    


# This function will augment the image for any given attribute from a given list of positive attribute image and the attribute names
def augment_images(data_root_dir, att, src_img_name, att_img_list):
    # The number of augmentations to be created for a given single image 
    n_tforms = len(att_img_list)  
    # n_tforms = 1 # Only creating a single flip copy of the original input image for pose attribute 
    
    for id in range(0, n_tforms): 
        att_img_name = att_img_list[id]

        # Extracting the segmentation mask for any given attribute edit. Some edits such as lips and eyes require two segmentation masks to be combined into one 
        flag_mask, segm = create_global_segm(data_root_dir, att_img_name, att)

        if (flag_mask == False):
            print("continuing ")
            continue
        # cv2.imshow('segm' + att, segm)
        # cv2.waitKey(2000)      

        # Now we can create augmentations for the input image and given attribute image with its combined segmentation mask 
        create_augmentations_wo_alignment(data_root_dir, src_img_name, att_img_name, segm, att, change_clr=False) 


# Utils function to select some augmentations which are manually selected
def select_samples(src_folder, dst_folder):
    # Eyeglass attribute: 
    # img_list = ['10427', '7962', '21755', '12331', '1299', '13684', '12430', '7091']
    # att_list = ['2686', '20158','4268','22909','15863','26665','19903','28838','20150','29908']

    # Style Hair attribute:
    img_list = ['1597', '14834', '28808', '20412', '843', '8260', '625', '25772', '1761', '5476', '6692', '751', '3469']
    att_list = ['12', '1597', '2499', '4153', '6864', '9590', '10097', '10550', '26993', '29230', '26816', '20160', '23250'] 

    for file_ in os.listdir(src_folder):
        for img_nm in img_list:
            # Copyieng the original image 
            if (img_nm + '.jpg' == file_):
                fp = os.path.join(src_folder, file_)
                shutil.copy(fp, dst_folder)

            # Copyieng selected image transformation into a separate folder  
            for att_nm in att_list:
                if (img_nm in file_ and att_nm in file_):
                    fp = os.path.join(src_folder, file_)
                    shutil.copy(fp, dst_folder)
            
    print("images copied. ")

# This function provides the functionality to augment all the images upto some number by various transformations from the attribute types 
def batch_augment_images():
    # np.random.seed(1)
    data_root_dir = '../CelebAMask-HQ/data_filtered/renew'  

    # Transforming the pose of the face, dataset can be any of attributes negative image set 
    """
    att = 'pose'
    src_df = pd.read_csv('../data_files/attribute_datasets/Eyeglasses_neg.csv')
    pose_img_df = src_df
    att_img_list = pose_img_df['file_name']
    """

    # Transforming eyeglass attribute 
    att = 'eye_g'
    src_df = pd.read_csv('../data_files/attribute_datasets/Eyeglasses_neg.csv')  
    eyeg_img_df = pd.read_csv('../data_files/attribute_datasets/Eyeglasses_pos.csv')
    att_img_list = eyeg_img_df['file_name']
    # att_img_list = ['888', '1565', '1598', '1817', '2686', '3824', '4081', '4289', '5220', '6114', '29778', '29995', 
    #                 '168', '1153', '1183', '1158', '1802', '4967', '1856', '3078', '3089', '4015', '4389', '6513']
    


    # Transforming hair attribute, this is for syle editing of hairs hence we we will concatenate
    # positives from all the hair attribute database files 
    att = 'hair'
    src_df = pd.read_csv('../data_files/attribute_datasets/Receding_Hairline_pos.csv')
    hair_img_df1 = pd.read_csv('../data_files/attribute_datasets/Bangs_pos.csv')
    # hair_img_df2 = pd.read_csv('../data_files/attribute_datasets/Black_Hair_pos.csv')
    # hair_img_df3 = pd.read_csv('../data_files/attribute_datasets/Brown_Hair_pos.csv')
    # hair_img_df4 = pd.read_csv('../data_files/attribute_datasets/Gray_Hair_pos.csv')
    hair_img_df5 = pd.read_csv('../data_files/attribute_datasets/Straight_Hair_pos.csv')
    hair_img_df6 = pd.read_csv('../data_files/attribute_datasets/Wavy_Hair_pos.csv')
    hair_img_df = pd.concat([hair_img_df1, hair_img_df5, hair_img_df6]) 
    att_img_list = hair_img_df['file_name']

    # Transforming hairs to bangs
    """
    att = 'bangs'
    src_df = pd.read_csv('../data_files/attribute_datasets/Receding_Hairline_pos.csv')  
    hair_img_df = pd.read_csv('../data_files/attribute_datasets/Bangs_pos.csv') 
    print("Bang image df head:") 
    print(hair_img_df.head())
    att_img_list = hair_img_df['file_name']
    print("att img list filtered: ", att_img_list)
    """


    # Transforming hairs to bangs
    """
    att = 'bald'  
    src_df = pd.read_csv('../data_files/attribute_datasets/Bald_pos.csv')  
    hair_img_df = pd.read_csv('../data_files/attribute_datasets/Black_Hair_pos.csv') 
    print("Bang image df head:") 
    print(hair_img_df.head())
    att_img_list = hair_img_df['file_name']
    """ 
    

    # Transforming hair attribute 
    """
    att = 'hair'
    src_df = pd.read_csv('../data_files/attribute_datasets/Straight_Hair_pos.csv')  
    hair_img_df = pd.read_csv('../data_files/attribute_datasets/Straight_Hair_neg.csv') 
    print("hair image df head:") 
    print(hair_img_df.head())
    att_img_list = list(hair_img_df[:25]['file_name'])  
    """

    # Transforming smile attribute 
    """
    att = 'smile'
    src_df = pd.read_csv('../data_files/attribute_datasets/Smiling_neg.csv')  
    smile_img_df = pd.read_csv('../data_files/attribute_datasets/Smiling_pos.csv')
    att_img_list = smile_img_df['file_name']
    """ 

    # Transforming eye attribute
    """
    att = 'eye'
    src_df = pd.read_csv('../data_files/attribute_datasets/Narrow_Eyes_neg.csv')
    eye_img_df = pd.read_csv('../data_files/attribute_datasets/Narrow_Eyes_pos.csv')
    att_img_list = list(eye_img_df[:25]['file_name']) 
    """ 

    
    # Transforming hat attribute 
    """
    att = 'hat'
    src_df = pd.read_csv('../data_files/attribute_datasets/Wearing_Hat_neg.csv')
    hat_img_df = pd.read_csv('../data_files/attribute_datasets/Wearing_Hat_pos.csv')
    att_img_list = hat_img_df['file_name'] 
    """ 

    n_tforms = 25
    n_imgs = 100

    # Sampling the attribute images first and keeping it fixed for all the varations 
    att_imgs = list(att_img_list.sample(n_tforms))  
    # Iterating over the source images to be augmented
    for j in range(0,n_imgs):
        print("Transforming image: {}".format(j))
        idx = np.random.randint(0, src_df.shape[0]) 
        src_img_name = src_df.iloc[idx]['file_name']  

        # Calling the augmentation algorithm for the given source image and all the various types of given attribute iamge set. 
        augment_images(data_root_dir, att, src_img_name, att_imgs)     

    print("transformed {} idx image".format(j))    

def main():
    # 2.1
    # batch_augment_images()    

    # 2.2
    src_folder = '../CelebAMask-HQ/data_filtered/renew/augmentations/hair' 
    dst_folder = '../CelebAMask-HQ/data_filtered/renew/augmentations/filtered_att_style_dataset/hair_style'
    select_samples(src_folder, dst_folder) 


if __name__ == "__main__":
    debug_flag = True
    main()