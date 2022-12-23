import model
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import SupportingFunctions as sf
import os
from skimage.metrics import structural_similarity as ssim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print('Test code has been started.')

### HYPERPARAMETERS
params = dict([('num_epoch', 200),
               ('batch_size', 1),
               ('learning_rate', 1e-3),
               ('num_workers', 0),          # It should be 0 for Windows machines
               ('exp_num', 7),              # CHANGE EVERYTIME
               ('save_flag', False),
               ('use_cpu', False),
               ('acc_rate', 4),
               ('K', 10)])   

### PATHS          
test_data_path  = 'Knee_Coronal_PD_RawData_392Slices_Test.h5'
test_coil_path  = 'Knee_Coronal_PD_CoilMaps_392Slices_Test.h5'
                   
# 0) Fix randomness for reproducible experiment
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# 1) Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and (not(params['use_cpu']))) else 'cpu')

# 2) Load Data
dataset = sf.KneeDataset(test_data_path, test_coil_path, params['acc_rate'], num_slice=10)
loaders, datasets= sf.prepare_test_loaders(dataset,params)
mask = dataset.mask.to(device)

####################################################
############## TEST CODE ###########################
####################################################
denoiser = model.ResNet().to(device)
denoiser.load_state_dict(torch.load('model_t__MOI_160.pt'))
denoiser.eval()
for i, (x0, xref, kspace, sens_map, rand_mask, index) in enumerate(loaders['test_loader']):
    with torch.no_grad():
        x0 = x0.to(device)
        xref = xref.to(device)
        sens_map = sens_map.to(device)
        kspace = kspace.to(device)
        # Forward pass
        xk = x0
        for k in range(params['K']):
            L, zk = denoiser(xk)
            xk = model.DC_layer(x0,zk,L,sens_map,mask)
            
        xc = model.DC_layer(x0,x0,0,sens_map,mask)
        
        xref = np.abs(xref.cpu().detach().numpy()[0,:,:])
        x0 = np.abs(x0.cpu().detach().numpy()[0,:,:])
        xc = np.abs(xc.cpu().detach().numpy()[0,:,:])
        xk = np.abs(xk.cpu().detach().numpy()[0,:,:])
        
        data_range=xref.max() - xref.min()
        ssim_0 = ssim(xref, x0, data_range=data_range)
        ssim_c = ssim(xref, xc, data_range=data_range)
        ssim_k = ssim(xref, xk, data_range=data_range)
        nmse_0 = sf.nmse(x0,xref)
        nmse_k = sf.nmse(xk,xref)
        nmse_c = sf.nmse(xc,xref)
        '''
        figure = plt.figure(figsize=(368/54.5,320/54.5))
        plt.imshow(x0,cmap='gray')
        plt.title(f'zero_filled_slice:{index.item():03d}', fontsize=20)
        ax = plt.gca()
        label = ax.set_xlabel('NMSE:'+f'{nmse_0:,.3f}'+'\n'+
                              'SSIM:'+f'{ssim_0:,.3f}', fontsize = 20)
        ax.xaxis.set_label_coords(0.17, 0.13)
        ax.xaxis.label.set_color('white')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.show()
        figure.savefig('x0'+f'_{i:03d}'+'.png')   
        
        figure = plt.figure(figsize=(368/54.5,320/54.5))
        plt.imshow(xc,cmap='gray')
        plt.title(f'CG-SENSE_slice:{index.item():03d}', fontsize=20)
        ax = plt.gca()
        label = ax.set_xlabel('NMSE:'+f'{nmse_c:,.3f}'+'\n'+
                              'SSIM:'+f'{ssim_c:,.3f}', fontsize = 20)
        ax.xaxis.set_label_coords(0.17, 0.13)
        ax.xaxis.label.set_color('white')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.show()
        figure.savefig('x'+f'_CG_{i:03d}'+'.png') 
        '''
        figure = plt.figure(figsize=(368/54.5,320/54.5))
        plt.imshow(xk,cmap='gray')
        plt.title(f'MOI_slice:{index.item():03d}', fontsize=20)
        ax = plt.gca()
        label = ax.set_xlabel('NMSE:'+f'{nmse_k:,.3f}'+'\n'+
                              'SSIM:'+f'{ssim_k:,.3f}', fontsize = 20)
        ax.xaxis.set_label_coords(0.17, 0.13)
        ax.xaxis.label.set_color('white')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.show()
        figure.savefig('x_MOI'+f'_{i:03d}'+'.png') 
        '''
        figure = plt.figure(figsize=(368/54.5,320/54.5))
        plt.imshow(xref,cmap='gray')
        plt.title(f'reference_slice:{index.item():03d}', fontsize=20)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.show()
        figure.savefig('xref'+f'_{i:03d}'+'.png')  
        '''
