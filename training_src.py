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
from test_cos import evaluation_AP_DICE_DAE

import torch
from torchvision.utils import save_image
import numpy as np

def train_with_DAE(run_name, ckp_path, results_path, encoder, bn, decoder, optimizer, device, bs=32, split='train', dataset='brats2021', iteration_epoch=32, epochs=2100, res=16, std=0.2, img_size=128):
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
    
    test_data = BrainDataset(
        dataset=dataset, 
        split='test', 
        n_tumour_patients=None, 
        n_healthy_patients=0
    )

    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = bs, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
    
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
            pseudo_anomaly = pseudo_anomaly.to(device)

            # forward
            inputs = encoder(Gaussian_img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, device=device, batch_size = bs, out_size=img_size)
            
            # visualize
            save_image(Gaussian_img, 'aug.png')
            save_image(org, 'img.png')
            save_image(pseudo_anomaly, 'anomaly_mask.png')
            save_image(anomaly_map, 'anomaly_map.png')
            
            # loss
            pseudo_anomaly = pseudo_anomaly.sum(dim=1, keepdim=True) 
            loss = loss_function(anomaly_map, pseudo_anomaly)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            loss_list.append(loss.item())
            
            if iter >= iteration_epoch:
                break
            
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 5 == 0:
            auroc_px, auroc_sp, ap, dice = evaluation_AP_DICE_DAE(run_name, encoder, bn, decoder, test_dataloader, device, epoch, img_size)
            print('Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Average Precision:{:.3f}, DICE:{:.3f}'.format(auroc_px, auroc_sp, ap, dice))
            
            # save the checkpoints
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict(),
                        'last_epoch': epoch}, ckp_path)
            
            # Write the rsults
            with open(results_path, 'a') as f:
                f.writelines('Epoch:{}, Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Average Precision:{:.3f}, DICE:{:.3f}\n'.format(epoch, auroc_px, auroc_sp, ap, dice))
    
    
  
    
if __name__ == '__main__':
    train_with_DAE()
    
    
    