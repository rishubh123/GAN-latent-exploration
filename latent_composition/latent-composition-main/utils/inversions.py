import torch
from collections import OrderedDict
from . import util, pbar
from torch.nn.functional import l1_loss, mse_loss
from . import losses 
from networks.psp import id_loss 
import torch.nn as nn
import sys

masked_l1_loss = losses.Masked_L1_Loss()
masked_lpips_loss = losses.Masked_LPIPS_Loss()

def invert_lbfgs(nets, target, mask=None, lambda_f=0.25, lambda_l=0.5,
                 num_steps=3000, initial_latent=None):   
    from . import LBFGS
    G = nets.generator
    E = nets.encoder 
    G.eval()
    E.eval()
    G.cuda()
    E.cuda()

    if mask is None:
        # mask is just all ones (no mask)
        mask = torch.ones_like(target)[:, :1, : :]

    torch.set_grad_enabled(False)

    true_x = target.cuda()
    mask = mask.cuda()
    init_z = nets.encode(true_x, mask)
    invertim = nets.decode(init_z) 
    
    if initial_latent is not None:
        assert lambda_f == 0.0
        init_z = initial_latent
    current_z = init_z.clone()
    target_x = target.clone().cuda()
    target_f = nets.encode(target.cuda(), mask)
    parameters = [current_z]

    util.set_requires_grad(False, G, E)
    util.set_requires_grad(True, *parameters)
    optimizer = LBFGS.FullBatchLBFGS(parameters)

    def compute_all_loss():
        current_x = nets.decode(current_z)
        all_loss = {}
        all_loss['x'] = ((1-lambda_l) * masked_l1_loss(current_x, target_x, mask) + lambda_l * masked_lpips_loss(current_x, target_x, mask))
        all_loss['z'] = 0.0 if not lambda_f else (
            mse_loss(target_f, nets.encode(current_x, mask)) * lambda_f)
        return current_x, all_loss

    def closure():
        optimizer.zero_grad()
        _, all_loss = compute_all_loss()
        return sum(all_loss.values())

    losses = []
    iterator = pbar(range(num_steps + 1))
    with torch.enable_grad():
        for step_num in iterator:
            if step_num == 0:
                loss = closure()
                loss.backward()
            else:
                options = {'closure': closure, 'current_loss': loss,
                        'max_ls': 10}
                loss, _, lr, _, _, _, _, _ = optimizer.step(options)
            losses.append(loss.detach().cpu().numpy())
    # get final results
    with torch.no_grad():
        current_x, all_loss = compute_all_loss()
    checkpoint_dict = OrderedDict(all_loss)
    checkpoint_dict['loss'] = sum(all_loss.values()).item()
    checkpoint_dict['init_z'] = init_z
    checkpoint_dict['target_x'] = target_x
    checkpoint_dict['invertim_x'] = invertim 
    checkpoint_dict['current_z'] = current_z
    checkpoint_dict['current_x'] = current_x
    return checkpoint_dict, losses



# Custom function for optimizing the latent to preserve the identity of the generated image 
identity_loss = id_loss.IDLoss().cuda().eval()
def optimize_latent_for_id(nets, origim, att_dir, strength, mask=None, num_steps=3000, initial_latent=None): 
    num_steps = 50 
    temp = 10
    print("optimizing for identity, for num steps: ", num_steps)
    # origim: Original image which has to be edited 
    # invertim: Inverted image for the original image  
    # transformim: Inverted image trasnfromed after applying the transformation  

    from . import LBFGS 
    G = nets.generator
    E = nets.encoder
    G.eval()
    E.eval()
    G.cuda()
    E.cuda() 

    # Initializing the image mask for encoding the input image 
    if mask is None:
    # mask is just all ones (no mask)
        mask = torch.ones_like(origim)[:, :1, : :]

    torch.set_grad_enabled(False)

    true_x = origim.cuda()   # Original image which will be inverted in the latent space 
    mask = mask.cuda()

    # Initialize z by first inverting the original image into the latent space and adding the masked direction vector for any attribute
    init_z = nets.encode(true_x, mask) 
    # Initializing the inverted image  by the encoded latent vector 
    invertim = nets.decode(init_z) 

    print("attribute direction shape: ", att_dir.shape) 
    # Initializing the latent mask to all ones, This latent is layer wise mask: single float for each layer.
    # There are 18 layers in stylegan model and their is a single latent mask value for each layer 
    latent_mask_raw = torch.ones((att_dir.shape[1], 1)).cuda() 
    print("latent mask raw shape: ", latent_mask_raw.shape) 

    # Applying softmax to make the distribution over layers 
    latent_mask = nn.Softmax(dim=0)(latent_mask_raw / temp) 
    # print("latent mask after softmax: ", latent_mask)  

    start_z = init_z + strength * (latent_mask * att_dir) # Element wise multiplication of the latent_mask and the attribute direction 

    # Learning the latent mask raw which will be passed onto softmax before being added 
    current_mask = latent_mask_raw.clone() 
    current_z = start_z.clone() 

    # Target image is the inverted image 
    target_id_x = invertim.clone().cuda() 
    # target_id_x = origim.clone().cuda() 

    # Defining the parameters to be optimized during the training process 
    # parameters = [latent_mask_raw]
    parameters = [current_mask] # directly optimizing the latent mask instead of the softmax input 
    # parameters = [current_z]     

    util.set_requires_grad(False, G, E)
    util.set_requires_grad(True, *parameters)
    optimizer = LBFGS.FullBatchLBFGS(parameters) 
    reshape = torch.nn.AdaptiveAvgPool2d((256, 256))  

    def compute_all_loss():
        # Forward pass
        latent_mask = nn.Softmax(dim=0)(current_mask / temp)   
        current_z = init_z + strength * (latent_mask * att_dir) 
        current_x = nets.decode(current_z)
        all_loss = {}

        # Computing the identity loss between the inverted image and the reconstrcuted image 
        loss_id, sim_improvement, id_logs = identity_loss(reshape(current_x), reshape(target_id_x), reshape(target_id_x)) 
        all_loss['x'] = loss_id
        
        return current_x, all_loss

    def closure():
        optimizer.zero_grad()
        _, all_loss = compute_all_loss()
        return sum(all_loss.values())

    
    # Optimization iterations to optimize for the best latent mask values
    losses = []
    iterator = pbar(range(num_steps + 1))
    print("initial value of the latent mask: ", latent_mask) 
    
    with torch.enable_grad():
        for step_num in iterator:
            if step_num == 0:
                loss = closure()
                loss.backward()
            else:
                options = {'closure': closure, 'current_loss': loss,
                        'max_ls': 10}
                loss, _, lr, _, _, _, _, _ = optimizer.step(options)
            losses.append(loss.detach().cpu().numpy())
    
    print("final value of the latent mask: ", nn.Softmax(dim=0)(current_mask / temp))  
    # print("total losses:", losses)


    # Applying softmax to make the distribution over layers 
    # latent_mask_f = nn.Softmax(dim=0)(latent_mask_raw) 
    # print("latent mask after training: ", latent_mask) 

    # get final results
    with torch.no_grad():
        current_x, all_loss = compute_all_loss()
    checkpoint_dict = OrderedDict(all_loss)
    checkpoint_dict['loss'] = sum(all_loss.values()).item()
    checkpoint_dict['init_z'] = init_z
    checkpoint_dict['invertim_x'] = invertim 
    checkpoint_dict['current_z'] = current_z
    checkpoint_dict['current_x'] = current_x
    return checkpoint_dict, losses
