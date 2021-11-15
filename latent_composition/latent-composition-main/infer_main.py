import torch
import numpy as np
import pandas as pd
from utils import show, renormalize, pbar
from utils import util, paintwidget, labwidget, imutil, inversions
from networks import networks
from PIL import Image
import os
from matplotlib.pyplot import imshow 
# import skvideo.io
from torchvision import transforms
import time  


def load_image_tensor(img_path, outdim): 
    transform = transforms.Compose([
            transforms.Resize(outdim), 
            transforms.CenterCrop(outdim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    

    source_im = transform(Image.open(img_path))[None].cuda()
    return source_im

# Function to perform forward pass to convert image to latent
def encode_forward(nets, outdim, img_path): 
    img_save_size = 256
    source_im = load_image_tensor(img_path, outdim)

    # Image to be saved in small size 
    save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
    # Performing a forward pass of the encoder
    with torch.no_grad():
        # out = nets.invert(source_im)
        
        mask = torch.ones_like(source_im)[:, [0], :, :]
        z = nets.encode(source_im, mask)
        print("z vector: ", z.shape)
        
    return z, save_src_img


# Function to perform forward pass of the decoder to convert the latent back to image space 
def decode_forward(nets, outdim, z):
    with torch.no_grad():
        out_s = nets.decode(z)

    return out_s


def load_nets():
    
    # bonus stylegan encoder trained on real images + identity loss
    # nets = networks.define_nets('stylegan', 'ffhq', ckpt_path='pretrained_models/sgan_encoders/ffhq_reals_RGBM/netE_epoch_best.pth')
    
    # stylegan trained on gsamples + identity loss
    nets = networks.define_nets('stylegan', 'ffhq')
    return nets

# Function to encode images into latent embeddings ans saving them in separate files 
def save_embeddings(nets, outdim, img_path_set, dst_path_set, embds_path_set):  
    # use a real image as source
    for i in range(0, len(img_path_set)):
        img_path = img_path_set[i] # Image path 
        img_dst_path = dst_path_set[i] # Path to save inverted image 
        embds_dst_path = embds_path_set[i] # Path to save the latent embeddings 
        img_save_size = 128

        source_im = load_image_tensor(img_path, outdim) 

        save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
                
        # Image.show(renormalize.as_image(source_im[0]).resize((256, 256), Image.LANCZOS))
        # show(['Source Image', renormalize.as_image(source_im[0]).resize((256, 256), Image.LANCZOS)])
        
        # Performing a forward pass of the encoder and the decoder
        with torch.no_grad():
            # out = nets.invert(source_im)
            
            mask = torch.ones_like(source_im)[:, [0], :, :]
            z = nets.encode(source_im, mask)
            out_s = nets.decode(z)  # Output of image inversion by two step encoding and decoding 

            print("z vector: ", z.shape)
            
            # Converting the latent into numpy vector 
            z_np = z.cpu().detach().numpy()
            print("saving latent: ", embds_dst_path)
            np.save(embds_dst_path, z_np) 

            # show(['GAN Reconstruction direct', renormalize.as_image(out[0]).resize((256, 256), Image.LANCZOS)])
            # show(['GAN Reconstruction stepwise', renormalize.as_image(out_s[0]).resize((256, 256), Image.LANCZOS)])

            save_out_s_img = renormalize.as_image(out_s[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

            combined_display_image = np.hstack([save_src_img, save_out_s_img]) 
            save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 
            
            print("saving image: ", img_dst_path)
            save_img.save(img_dst_path) 


# Extract embeddings for all the images in the dataset. This function calls save_embeddings() function internally 
def extract_latents():
    outdim = 1024 # For faces 
    nets = load_nets()
    root_path = '../CelebAMask-HQ/' 

    # Defining the source folder for reading the input images 
    imgs_path = os.path.join(root_path, 'data_filtered/img')    
    dst_root_path = os.path.join(root_path, 'data_filtered/inversion')
    embds_root_path = os.path.join(root_path, 'data_filtered/latents')

    # Defining the file paths as a list to read and save the embeddings 
    img_names = [img for img in os.listdir(imgs_path)]
    dst_path_set = [os.path.join(dst_root_path, img) for img in img_names]
    imgs_path_set = [os.path.join(imgs_path, img) for img in img_names]
    embds_path_set = [os.path.join(embds_root_path, img[:-4]+'.npy') for img in img_names]  

    # Number of samples to be processes
    # n_samples = 2

    imgs_path_set = imgs_path_set[:] # Currently reading only 10 images for inference 
    dst_path_set = dst_path_set[:]
    ebmds_path_set = embds_path_set[:]

    print("image set: ", imgs_path_set) 
    save_embeddings(nets, outdim, imgs_path_set, dst_path_set, ebmds_path_set)  


# This function will compute particular dominant dirs for the given attributes and save the estimated direction into a numpy array. 
def compute_single_direction(att, data_files_root, embds_path_root, n):
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


# Computing directions for each of the atrributes given to the network
def compute_all_directions():  
    atts_list = ['eyeglasses', 'hat', 'smile']
    data_files_root = '../data_files'
    embds_path_root = '../CelebAMask-HQ/data_filtered/latents'

    n = 50 # Number of samples for each pos and neg categories 
    for i in range(0, len(atts_list)):
        compute_single_direction(atts_list[i], data_files_root, embds_path_root, n)



def edit_image(nets, img_path, img_idx, img_transform_path, atts_list, latent_paths):
    outdim = 1024
    img_save_size = 256 
    
    for i in range(0, len(atts_list)):
        att = atts_list[i]
        latent_save_path = latent_paths[i] 

        print("loading att latent dir file: ", latent_save_path)
        latent_dir = np.load(latent_save_path)
        latent_dir_tensor = torch.from_numpy(latent_dir).cuda()

        z, save_src_img = encode_forward(nets, outdim, img_path)

        alpha = 2.0
        zT = z + alpha*latent_dir_tensor

        out_z_img = decode_forward(nets, outdim, z)
        out_zT_img = decode_forward(nets, outdim, zT) 

        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
        save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

        combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
        save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 

        fn = img_idx + '_transformed_nc2_50_' + att + '_' + str(alpha) + '.jpg'
        save_img_path = os.path.join(img_transform_path, fn)

        print("saving image: ", save_img_path)
        save_img.save(save_img_path) 



def edit_image_identity(nets, img_path, img_idx, img_transform_path, atts_list, latent_paths):
    outdim = 1024
    img_save_size = 256
    
    for i in range(0, len(atts_list)):
        att = atts_list[i]
        latent_save_path = latent_paths[i] 

        print("loading att latent dir file: ", latent_save_path)
        latent_dir = np.load(latent_save_path)
        latent_dir_tensor = torch.from_numpy(latent_dir).cuda()
        att_strength = 2.0 

        source_im = load_image_tensor(img_path, outdim) 

        # Performing the latent optimization to estimate the identity preserved direction 
        checkpoint_dict, opt_losses = inversions.optimize_latent_for_id(nets, source_im, latent_dir_tensor, att_strength)
        
        # Image to be saved in small size 
        save_src_img = renormalize.as_image(source_im[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 
        out_z_img = checkpoint_dict['invertim_x'].detach().clone()
        out_zT_img = checkpoint_dict['current_x'].detach().clone() 

        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
        save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

        combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
        save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 

        fn = img_idx + '_transformed_' + att + '_' + str(att_strength) + '.jpg'
        save_img_path = os.path.join(img_transform_path, fn)

        print("saving image: ", save_img_path)
        save_img.save(save_img_path)



def edit_image_set(): 
    nets = load_nets()
    img_path_root = '../CelebAMask-HQ/data_filtered/test_imgs'
    img_transform_path = '../CelebAMask-HQ/data_filtered/transform_imgs_test_nc2'
    data_files_root = '../data_files'

    img_idxs = [img for img in os.listdir(img_path_root)]
    img_paths = [os.path.join(img_path_root, img_id) for img_id in img_idxs]

    atts_list = ['eyeglasses', 'hat', 'smile'] 
    latent_paths = ['avg_latent_dir_nc2_50_eyeglasses.npy', 'avg_latent_dir_nc2_50_hat.npy', 'avg_latent_dir_nc2_50_smile.npy']

    latent_paths = [os.path.join(data_files_root, lp) for lp in latent_paths]

    # Number of images to be processed 
    n = 15
    for i in range(0, n): 
        edit_image(nets, img_paths[i], img_idxs[i], img_transform_path, atts_list, latent_paths) 


if __name__ == "__main__":  
  print("running main ...")
  # extract_latents()
  # print("computing all directions")
  # compute_all_directions() 
  print("editing image dirs")
  edit_image_set()
 