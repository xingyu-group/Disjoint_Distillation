import torch
# from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
# from dataset import MVTecDataset
from dataloader_zzx import MVTecDataset, Medical_dataset, MVTecDataset_cross_validation

import torch.backends.cudnn as cudnn
import argparse
from test_cos import evaluation
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms
    
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


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(args):
    print(args)
    epochs = 200
    learning_rate = 0.005
    # batch_size = 16   
    image_size = 256
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    main_path = '/home/zhaoxiang/dataset/{}'.format(args.dataset_name)
    run_name = args.experiment_name + '_' + args.dataset_name + '_' + args.augmentation_method
    
    
    dirs = os.listdir(main_path)
    
    for dir_name in dirs:
        if 'train' in dir_name:
            train_dir = dir_name
        elif 'test' in dir_name:
            if 'label' in dir_name:
                label_dir = dir_name
            else:
                test_dir = dir_name
    if 'label_dir' in locals():
        dirs = [train_dir, test_dir, label_dir]   

    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    ckp_folder = os.path.join('/home/zhaoxiang/output', run_name)
    if not os.path.exists(ckp_folder):
        os.mkdir(ckp_folder)
    ckp_path = os.path.join(ckp_folder, 'last.pth')    
    results_path = os.path.join(ckp_folder, 'results.txt')    
    
    
    test_transform, _ = get_data_transforms(args.img_size, args.img_size)
    train_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    val_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name, args = args)
    test_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name, args = args, rgb=True)
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.bs, shuffle = False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    
    encoder = torch.nn.DataParallel(encoder, device_ids=[0, 1])
    bn = torch.nn.DataParallel(bn, device_ids=[0, 1])
    decoder = torch.nn.DataParallel(decoder, device_ids=[0, 1])
    
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        # for img, label in train_dataloader:
        # auroc_px, auroc_sp, aupro_px = evaluation(run_name, encoder, bn, decoder, test_dataloader, device, epoch)
        
        for img, aug, anomaly_mask in tqdm(train_dataloader):
            
            img = torch.reshape(img, (-1, 1, args.img_size, args.img_size))
            aug = torch.reshape(aug, (-1, 1, args.img_size, args.img_size))
            anomaly_mask = torch.reshape(anomaly_mask, (-1, 1, args.img_size, args.img_size))
            
            # aug = aug.expand(3,*aug.shape[1:])
            aug = torch.cat([aug, aug, aug], dim=1)
            # img = img.to(device)
            aug = aug.to(device)
            anomaly_mask = anomaly_mask.to(device)
            
            inputs = encoder(aug)
            outputs = decoder(bn(inputs))
            
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, device=device, batch_size=args.bs)
            
            # Visualize the results
            save_image(aug, 'aug.png')
            save_image(anomaly_mask, 'anomaly_mask.png')
            save_image(anomaly_map, 'anomaly_map.png')
            
            loss = loss_function(anomaly_map, anomaly_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp = evaluation(run_name, encoder, bn, decoder, test_dataloader, device, epoch)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}'.format(auroc_px, auroc_sp))
            
            # save the checkpoints
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
            
            # Write the rsults
            with open(results_path, 'a') as f:
                f.writelines('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}\n'.format(auroc_px, auroc_sp))
    return auroc_px, auroc_sp




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    parser.add_argument('--experiment_name', default='Disjoint_Distillation', choices=['DRAEM_Denoising_reconstruction, liver, brain, head'], action='store')
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--colorRange', default=100, action='store')
    parser.add_argument('--threshold', default=200, action='store')
    parser.add_argument('--number_iterations', default=1, action='store')
    parser.add_argument('--rejection', default=False, action='store')
    parser.add_argument('--control_texture', default=False, action='store')
    parser.add_argument('--cutout', default=False, action='store')
    
    # take care every time
    parser.add_argument('--dataset_name', default='Brain_MRI', choices=['hist_DIY', 'Brain_MRI', 'CovidX', 'RESC_average', 'BraTs'], action='store')
    parser.add_argument('--augmentation_method', default= 'gaussian_noise', choices=['Gaussian_noise', 'Cutpaste', 'random_shape', 'RESC_average', 'BraTs'], action='store')
    
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument('--gpu_id', default=['0','1'], action='store', type=str, required=False)
    
    
    
    args = parser.parse_args()
    

    setup_seed(111)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpu_id is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpu_id)):
            gpus = gpus + args.gpu_id[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    with torch.cuda.device(args.gpu_id):
        train(args)

