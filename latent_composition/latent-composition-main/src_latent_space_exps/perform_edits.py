"""
Date: 29/12/2021
This module has the functionality of creating the desired attribute edits for any given input image.
We leverage the latent direction estimated previously and use this direction to edit any new input image.
Specifically, input image is first encoded using the StyleGAN2 encoder into a latent code w. This latent code
is then transformed to obtain a new latent code w' which is then finally used to synthesize an image using StyleGAN2 generator model.
"""



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
 

# Edit image function for generating group of image results with various alpha values 
def edit_image_group(nets, img_paths, img_idxs, img_transform_path, att, latent_path):
    outdim = 1024
    img_save_size = 256 

    print("loading att latent dir file: ", latent_path)
    latent_dir = np.load(latent_path)
    latent_dir_tensor = torch.from_numpy(latent_dir).cuda()    

    # number of images to be edited 
    n = 5
    image_matrix = []
    for i in range(11, 11+n):
        img_path = img_paths[i]
        z, save_src_img = encode_forward(nets, outdim, img_path)

        alphas = [0.6, 0.8, 1.0, 1.5] 
        image_column = []

        # Inverted image
        out_z_img = decode_forward(nets, outdim, z)
        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
        image_column.append(save_out_z_img)

        for alpha in alphas:
            # print("using alpha value of:", alpha)
            zT = z + alpha*latent_dir_tensor

            # Transformed image 
            out_zT_img = decode_forward(nets, outdim, zT) 
            save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

            image_column.append(save_out_zT_img)

        image_column = np.vstack(image_column)
        image_matrix.append(image_column) 
        # print("image column shape: ", image_column.shape)

    group_image = np.hstack(image_matrix)
    print("group image shape: ", group_image.shape)
    save_img = Image.fromarray(np.uint8(group_image)).convert('RGB') 
    fn = 'report_transform_imgs_test_nc2' + att + '_' + str(alpha) + '.jpg'
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
        att_strength = 40.0

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

        fn = img_idx + '_transformed_nc2_50_id_' + att + '_' + str(att_strength) + '.jpg'
        save_img_path = os.path.join(img_transform_path, fn)

        print("saving image: ", save_img_path)
        save_img.save(save_img_path)


# computing average direction based on the interpolation weights for latents 
def compute_intepolate_dir(latents, weights):
    store = np.zeros(latents[0].shape, dtype = np.float16)
    for i in range(0, len(weights)):
        store += latents[i] * weights[i]              

    return store

# Editing images by performing interpolations between different attribute types to create a new attribute 
def edit_image_interpolate_atts(nets, img_path, img_idx, img_transform_path, latent_paths): 
    outdim = 1024
    img_save_size = 256 
    scale_factor = 18

    latents = []
    for lp in latent_paths:
        lat = np.load(lp)
        lat_leng = np.linalg.norm(lat)
        lat = scale_factor * lat/lat_leng
        print("lat leng: ", lat_leng)
        latents.append(lat)


    basis_latents = [latents[0] - latents[1],
                     latents[0] - latents[2],
                     latents[1] - latents[3],
                     latents[6] - latents[1],
                     latents[0] - latents[7],
                     latents[6] - latents[2],
                     latents[3] - latents[8]]

    dc_shift = latents[0]

    """
    interpolation_values = [[1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.8, 0.2, 0.0, 0.0, 0.0],
                             [0.2, 0.8, 0.0, 0.0, 0.0],
                             [0.0, 0.5, 0.5, 0.0, 0.0],
                             [0.5, 0.0, 0.5, 0.0, 0.0],
                             [0.0, 0.5, 0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.5, 0.5, 0.0]] 
    """
    # exit()

    basis_coeffs = [[round(np.random.uniform(-0.35, 0.35),2) for i in range(0,7)] for j in range(0,10)]
    print("basis coeffs: ", basis_coeffs)
    
    # Iterating over the set of all the interpolation values to be used for averaging 
    for i in range(0, len(basis_coeffs)):
        weights = basis_coeffs[i]

        # Computing the average latent with the given set of weights 
        avg_latent_dir = compute_intepolate_dir(basis_latents, weights) 

        avg_latent_dir = avg_latent_dir + dc_shift
        # print("norm length of average latent: ", np.linalg.norm(avg_latent_dir))
        # exit()

        latent_dir_tensor = torch.from_numpy(avg_latent_dir).cuda() 
        # print("latent avg type: ", type(avg_latent_dir[0]), "single latent type: ", type(latents[0][0]))
        # exit()

        z, save_src_img = encode_forward(nets, outdim, img_path)

        alpha = 0.8
        zT = z + alpha*latent_dir_tensor  

        out_z_img = decode_forward(nets, outdim, z)
        out_zT_img = decode_forward(nets, outdim, zT)  

        save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
        save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)

        combined_display_image = np.hstack([save_src_img, save_out_z_img, save_out_zT_img])
        save_img = Image.fromarray(np.uint8(combined_display_image)).convert('RGB') 

        fn = img_idx + '_transformed_interpolate_atts_average5_norm_basis_eg' + str(weights) + '_w_' + str(alpha) + '.jpg'
        folder_name = os.path.join(img_transform_path, str(alpha))
        
        if (not os.path.exists(folder_name)):
            os.mkdir(folder_name)

        save_img_path = os.path.join(folder_name, fn) 

        print("saving image: ", save_img_path)
        save_img.save(save_img_path) 



# Editing images by performing interpolations between different attribute types to create a new attribute 
def edit_image_interpolate_atts_group(nets, img_path, img_idx, img_transform_path, latent_paths): 
    outdim = 1024
    img_save_size = 256 
    scale_factor = 18           

    latents = []
    for lp in latent_paths:
        lat = np.load(lp)
        lat_leng = np.linalg.norm(lat)
        lat = scale_factor * lat/lat_leng
        # print("lat leng: ", lat_leng)
        latents.append(lat)


    basis_latents = [latents[0] - latents[1],
                     latents[0] - latents[2],
                     latents[1] - latents[3],
                     latents[6] - latents[1],
                     latents[0] - latents[5], # 7
                     latents[6] - latents[2],
                     latents[3] - latents[8]]

    dc_shift = latents[0]
    # exit()

    basis_coeffs = [[round(np.random.uniform(-0.35, 0.35),2) for i in range(0,7)] for j in range(0,10)]
    # print("basis coeffs: ", basis_coeffs)

    image_matrix = []
    # Source image decode latent code 
    z, save_src_img = encode_forward(nets, outdim, img_path)
    out_z_img = decode_forward(nets, outdim, z)
    save_out_z_img = renormalize.as_image(out_z_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS) 

    # Creating additional first row just to cover only single input source image
    first_row = np.ones((256, 10*256, 3)) * 255
    first_row[:256, :256, :] = save_out_z_img 
    image_matrix.append(first_row) 

    alphas = [0.8, 1.0, 1.2]    
    for alpha in alphas:
        # Iterating over the set of all the interpolation values to be used for averaging 
        image_row = []

        for i in range(0, len(basis_coeffs)):
            weights = basis_coeffs[i]

            # Computing the average latent with the given set of weights 
            avg_latent_dir = compute_intepolate_dir(basis_latents, weights) 
            avg_latent_dir = avg_latent_dir + dc_shift 

            latent_dir_tensor = torch.from_numpy(avg_latent_dir).cuda() 

            zT = z + alpha*latent_dir_tensor  

            out_zT_img = decode_forward(nets, outdim, zT)  
            save_out_zT_img = renormalize.as_image(out_zT_img[0]).resize((img_save_size, img_save_size), Image.LANCZOS)
            image_row.append(save_out_zT_img)  

        image_row = np.hstack(image_row)
        image_matrix.append(image_row)  

    image_matrix = np.vstack(image_matrix)
    save_img = Image.fromarray(np.uint8(image_matrix)).convert('RGB') 

    fn = img_idx[:-4] + 'report_transform_imgs_test_att_interpolate_norm_basis' + '.jpg'
    save_img_path = os.path.join(img_transform_path, fn) 

    print("saving image: ", save_img_path)
    save_img.save(save_img_path) 


# Editing image set with the saved latents [Vanilla], applying transformation learnt by pairwise imgs and non-paired images 
def edit_image_set(): 
    nets = load_nets()
    img_path_root = '../CelebAMask-HQ/data_filtered/test_imgs'
    img_transform_path = '../CelebAMask-HQ/data_filtered/transform_imgs_test_nc2_id'
    data_files_root = '../data_files'

    img_idxs = [img for img in os.listdir(img_path_root)]
    img_paths = [os.path.join(img_path_root, img_id) for img_id in img_idxs]

    atts_list = ['eyeglasses', 'hat', 'smile'] 
    latent_paths = ['avg_latent_dir_nc2_50_eyeglasses.npy', 'avg_latent_dir_nc2_50_hat.npy', 'avg_latent_dir_nc2_50_smile.npy']

    latent_paths = [os.path.join(data_files_root, lp) for lp in latent_paths]

    # Number of images to be processed 
    n = 5
    print("Editing {} images".format(n)) 

    # Module for performing image edit in a batch and saving a single png image
    # img_transform_path = '../CelebAMask-HQ/data_filtered/report_transform_imgs_test_synthesized_5pairs'
    # atts_list = ['eyeglasses', 'hat']
    # latent_paths = ['avg_latent_dir_pairwise_5_eyeglasses_synth_aug.npy', 'avg_latent_dir_pairwise_5_hat_synth_aug.npy']
    # latent_paths = [os.path.join(data_files_root, lp) for lp in latent_paths]

    # for i in range(0, len(atts_list)): 
    #     edit_image_group(nets, img_paths, img_idxs, img_transform_path, atts_list[i], latent_paths[i]) 

    for i in range(0, n): 
        # edit_image(nets, img_paths[i], img_idxs[i], img_transform_path, atts_list, latent_paths) 
        edit_image_identity(nets, img_paths[i], img_idxs[i], img_transform_path, atts_list, latent_paths)

# Editing images with the .pkl latent directions estimated through various augmentations of the inputs 
def edit_image_set_interpolate_atts(): 
    nets = load_nets()
    img_path_root = '../CelebAMask-HQ/data_filtered/test_imgs'
    img_transform_path = '../CelebAMask-HQ/data_filtered/transform_imgs_test_interpolate_atts_average5_norm_basis_eg'  
    data_files_root = '../data_files' 

    img_idxs = [img for img in os.listdir(img_path_root)]
    img_paths = [os.path.join(img_path_root, img_id) for img_id in img_idxs]

    latent_dirs_pkl = 'all_latent_dirs_pairwise5_8167.pkl'
    # latent_dirs_pkl = 'all_latent_dirs_pairwise5_27995.pkl'
    latent_path = os.path.join(data_files_root, latent_dirs_pkl) 

    latent_paths = ['888','1565','2686','4081','4289','5220','6114','29778','29995']
    latent_paths = [lp + '__eye_g_transform_average_5_imgs_eyeglasses.npy' for lp in latent_paths]
    latent_paths = [os.path.join(data_files_root, lp) for lp in latent_paths]
    

    # Number of images to be processed 
    n = 10
    print("Editing {} images".format(n)) 

    # Images saving for multiple variations for a single image and creating a matrix collage out of it
    img_transform_path = img_transform_path = '../CelebAMask-HQ/data_filtered/report_transform_imgs_test_att_interpolate_norm_basis/'  

    for i in range(4, 4+n): 
        # edit_image_interpolate_atts(nets, img_paths[i], img_idxs[i], img_transform_path, latent_paths)  
        edit_image_interpolate_atts_group(nets, img_paths[i], img_idxs[i], img_transform_path, latent_paths)



if __name__ == "__main__":  
  print("running main ...")
  print("editing image dirs ...")  
  # edit_image_set()

  print("editing image with interpolation of attributes") 
  edit_image_set_interpolate_atts()   