import numpy as np
import cv2
import pandas as pd
import math
import os 


def display_landmarks(img, landmarks):
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

# Function to create augmentation of applyieng spectacles to a given input image 
def create_augmentation(data_root_dir, src_img_name, att_img_name, att, landmarks_df):
    src_img_path = os.path.join(data_root_dir, 'img', src_img_name) # Src image on which we will augment different attributes
    att_img_path = os.path.join(data_root_dir, 'img', att_img_name) # Attribute image from which we will extract visual attributes like eye_glasses or hat

    mask_path_root = str(format(int(att_img_name[:-4]), '05d')) # All the masks are saved in the format of 000xx.png
    mask_path = os.path.join(data_root_dir, 'anno-mask', mask_path_root + att + '.png') # Mask for the attribute image 

    if (not os.path.exists(mask_path)):
        return 

    # Image names for saving 
    image_save_name = os.path.join(data_root_dir, 'augmented', src_img_name[:-4] + '_' + att_img_name[:-4] + '_' + att + '.jpg')
    src_img_save_path = os.path.join(data_root_dir, 'augmented', src_img_name)

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

    cv2.imwrite('temp/sample_'+src_img_name[:-4] + att_img_name[:-4] + '.png', combined_imgs)
    cv2.imwrite(image_save_name, modified_img)


def transform_images():
    data_root_dir = '../CelebAMask-HQ/data_filtered/'
    
    att_df = pd.read_csv('../data_files/hat_pos_500.csv')
    src_df = pd.read_csv('../data_files/hat_neg_500.csv')

    landmarks_df = pd.read_csv('../data_files/landmarks_detected.csv') 

    att = '_hat'
    print("transforming hat")

    n_imgs = 100
    n_tforms = 5

    for j in range(0,n_imgs):
        idx = np.random.randint(0, src_df.shape[0])
        src_img_name = src_df.iloc[idx]['file_name']
        for i in range(0,n_tforms): 
            idy = np.random.randint(0, att_df.shape[0])
            att_img_name = att_df.iloc[idy]['file_name']
            create_augmentation(data_root_dir, src_img_name, att_img_name, att, landmarks_df)
        print("transformed {} idx image".format(j)) 


def main():
    transform_images()


if __name__ == "__main__":
    main()