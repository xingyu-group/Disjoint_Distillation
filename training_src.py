"""reimplement the DAE training loop
# 1. load data with the DAE dataset and dataloader
# 2. concat 3 modalities (originally 4).
# 3. Add gaussian random noise
# 4. implement disjoint distillation
5. Add testing Schemes
5. Added validation steps
5. (optional) make the number of modalities adjustable
"""

from data_DAE import BrainDataset
from dataloader_zzx import add_Gaussian_noise

from utils import cal_anomaly_map, loss_function

import torch

def train_with_DAE(encoder, bn, decoder, optimizer, device, bs=32, split='train', dataset='brats2021', iteration_epoch=32, epochs=2100, res=16, std=0.2, img_size=128):
    # dd = BrainAEDataDescriptor(dataset="brats2021", n_train_patients=None, n_val_patients=None,
    #                            seed=seed, batch_size=batch_size)
    
    # Initialize the training dataset
    train_data = BrainDataset(
        split='train', 
        dataset=dataset, 
        n_tumour_patients=0,
        n_healthy_patients=None,
        seed=0
    )
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = bs, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
    
    # Start the training loop
    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for iter, batch in enumerate(train_dataloader):
            # per iteration training optimization
            
            # data
            org = batch[0]
            # Retrieve 3 channel
            org = org[:,[0, 1, 3],:,:]
            
            Gaussian_img, pseudo_anomaly, pseudo_anomaly_mask = add_Gaussian_noise(org, res, std, img_size)  
            
            # to device
            Gaussian_img = Gaussian_img.to(device)
            pseudo_anomaly_mask = pseudo_anomaly_mask.to(device)

            # forward
            inputs = encoder(Gaussian_img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, device=device, batch_size = bs, out_size=img_size)
            
            # loss
            loss = loss_function(anomaly_map, pseudo_anomaly_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            loss_list.append(loss.item())
            
            if iter >= iteration_epoch:
                break
            
    
    
  
    
if __name__ == '__main__':
    train_with_DAE()
    
    
    