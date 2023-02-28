import os
from torch.nn import functional as F
import torch
import numpy as np
import random


join = os.path.join
def count_num_files(dir_path):
    files = os.listdir(dir_path)
    return len(files)
    
   
def cal_anomaly_map(fs_list, ft_list, device, batch_size=8, out_size=256, amap_mode='mul'):
    if amap_mode == 'mul':
        # anomaly_map = np.ones([out_size, out_size])
        anomaly_map = torch.ones([fs_list[0].shape[0], 1, out_size, out_size]).to(device)
    else:
        # anomaly_map = np.zeros([out_size, out_size])
        anomaly_map = torch.zeros([fs_list[0].shape[0], 1, out_size, out_size]).to(device)
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        # a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        # a_map = a_map[0,0,:,:]
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # for item in len(a):
    loss = torch.mean(1-cos_loss(a.view(a.shape[0],-1),
                                      b.view(b.shape[0],-1)))
    return loss
    
    
    
