import os
import numpy as np
from scipy.spatial import distance
import torch 
from PIL import Image 
import cv2 

from fid_module import *
from stylegan_utils import load_nets
from utils import renormalize
import face_recognition # Importing the face recognition library 

import sys 
sys.path.append('../') 


# This function will compute the fid for generated edited images given two set of folders or GAN generated images and edited images. 
def estimate_fid(fld1, fld2):
    batch_size = 32
    dims = 2048
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    fid_val = calculate_fid_given_paths([fld1, fld2], batch_size, device, dims) 

    return fid_val 

# This function will compute the face recognition scores for the images with same name from the given two folders 
def compare_face_id(src_folder, dst_folder):
    # Iterating over images in a given folder 
    eucds = []
    coss= []
    accs = []
    idx = 0
    for fn_ in os.listdir(src_folder):
        if (idx > 10):
            break
        orig_img_pt = os.path.join(src_folder, fn_)
        edit_img_pt = os.path.join(dst_folder, fn_)
        # print("comparing identity for: ")
        # print(orig_img_pt)
        # print(edit_img_pt) 

        orig_img = face_recognition.load_image_file(orig_img_pt)
        edit_img = face_recognition.load_image_file(edit_img_pt)

        orig_encoding = face_recognition.face_encodings(orig_img)[0]
        edit_encoding = face_recognition.face_encodings(edit_img)[0]

        cosine = distance.cosine(orig_encoding, edit_encoding)
        eucd = np.linalg.norm(orig_encoding-edit_encoding)

        results = face_recognition.compare_faces([orig_encoding], edit_encoding)
        
        eucds.append(eucd)
        coss.append(cosine) 
        accs.append(results)
        print(eucds, coss, accs)
        exit()
        idx += 1

    eucds = np.array(eucds)
    coss = np.array(coss)
    accs = np.array(accs)

    print("eucds: ", eucds.mean())
    print("coss: ", coss.mean())
    print("accs all: ", accs)





# This function will sample a number of z latents and generate the images corresponding to it using the StyleGAN2 model. 
def generate_real_imgs(dst_folder, n):
    outdim = 1024
    nets = load_nets()

    zs = nets.sample_zs(10*n, 0)
    seeds = np.random.choice(len(zs), n)

    print("zs shape: ", zs.shape)
    source_zs = zs[seeds]

    # Truncation factor for generating real images having realistic look 
    tr_factor = 0.7
    mean_zs = source_zs.mean(axis=0).reshape(-1,512)
    # print("mean zs shape: ", mean_zs.shape) 

    for id in range(0, source_zs.shape[0]):
        # img_raw = nets.zs2image(zs[id:id+1, :])

        # Generating real images with truncation factor 
        img_raw = nets.zs2image_truncated(zs[id:id+1], mean_zs, tr_factor)


        # img_raw = source_ims[id, ...]
        img = renormalize.as_image(img_raw[0]).resize((outdim, outdim), Image.LANCZOS) 
        img_dst_path = os.path.join(dst_folder, str(id) + '.jpg')

        save_img = Image.fromarray(np.array(img, np.uint8)).convert('RGB') 
        if (id % 100 == 0 or id == n-1):
            print("Saving the image at: ", img_dst_path) 
        save_img.save(img_dst_path)  

# This function will create hierarchical folder structure required for saving individual image files 
def make_folder(src_fld, method_types, img_types): 
    for mt in method_types:
        mfld_path = os.path.join(src_fld, mt)
        
        if (not os.path.exists(mfld_path)):
            os.mkdir(mfld_path)
        
        for it in img_types:
            fld_path = os.path.join(src_fld, mt, it)
            if (not os.path.exists(fld_path)):
                print("Creating a folder at path: ", fld_path)
                os.mkdir(fld_path) 


# This function will iterate over all the combined matrix images in the src folder and will save them separately into folders
def process_combined_images(src_folder, dst_folder): 
    #  iter -> IF/GS/SF/Ours -> Source/Edit1/Edit2/Edit3/Edit4
    # seq -> IF/GS/SF/Ours -> Source/Edit1/Edit2/Edit3/Edit4 

    method_types = ['IntefaceGAN', 'GanSpace', 'StyleFlow', 'Ours']
    img_types = ['source', 'edit1', 'edit2', 'edit3', 'edit4']

    make_folder(dst_folder, method_types, img_types)

    for im_ in os.listdir(src_folder):
        im_path = os.path.join(src_folder, im_)
        img_matrix = cv2.imread(im_path)

        imdim = 1024
        n_rows = int(img_matrix.shape[0] / imdim)
        n_cols = int(img_matrix.shape[1] / imdim)

        print("n rows: ", n_rows)
        print("n cols: ", n_cols)

        # Traversing over all the images and saving only the corresponding patch
        for r in range(0, n_rows):
            for c in range(0, n_cols):
                img_patch = img_matrix[r*imdim:(r+1)*imdim, c*imdim:(c+1)*imdim, :]   # Taking only the (r,c)th subimage to be cropped 
                img_dst_path = os.path.join(dst_folder, method_types[r], img_types[c], im_)
                cv2.imwrite(img_dst_path, img_patch) 

        print("Processed image: ", im_)



def run_main():
    """
    dst_folder = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/quant/gen_real_trunc/'
    print("performing generated real images sampled from latent space .. ") 
    generate_real_imgs(dst_folder, 1000)
    """

    # --------------------- Splitting the stack images into separate folder for easy computation -------------------------- # 
    """
    src_folder = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/compare_matrix_iter/'  
    dst_folder = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/quant/iter/'
    process_combined_images(src_folder, dst_folder)
    """

    """
    print("Computing fid for two folders ... ")

    folder1 = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/quant/gen_real_trunc/'
    folder2 = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/quant/seq/Ours/edit4' 
    fid = estimate_fid(folder1, folder2)
    print("computed fid value ours: ", fid) 
    """

    print("Computing identity score ...")
    orig_folder = '../../CelebAMask-HQ/data_filtered/renew/comparison_results/quant/seq/Ours/source'
    edit_folder =  '../../CelebAMask-HQ/data_filtered/renew/comparison_results/quant/seq/IntefaceGAN/edit3'

    compare_face_id(orig_folder, edit_folder)

if __name__ == "__main__":
    run_main()


# Sequential Editing 
# InterfacGAN: edit3-62.58, edit4-67.19
# StyleFlow: edit3-64.15, edit4-68.57
# GanSpace: edit3-57.15, edit4-59.90
# Ours: edit3- 44.89, edit4-56.79

# Sequential Editing, comparison with truncated generated images 
# InterfaceGAN: edit2-40.56, edit3-43.07, edit4-58.03
# StyleFlow: edit2-34.31,    edit3-47.81, edit4-50.08
# GanSpace: edit2-38.68,    edit3-42.38, edit4-43.55           # 36.33
# Ours: edit2-32.84,         edit3-34.59, edit5-56.95 