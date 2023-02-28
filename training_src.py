"""reimplement the DAE training loop
1. load data with the DAE dataset and dataloader
2. concat 3 modalities (originally 4).
3. Add gaussian random noise
4. implement disjoint distillation
5. (optional) make the number of modalities adjustable
"""

from data_DAE import BrainDataset

def train_with_DAE(split='train', dataset='brats2021'):
    # dd = BrainAEDataDescriptor(dataset="brats2021", n_train_patients=None, n_val_patients=None,
    #                            seed=seed, batch_size=batch_size)
    
    # Initialize the training dataset
    dataset = BrainDataset(
        split=split, 
        dataset=dataset, 
        n_tumour_patients=0,
        n_healthy_patients=None,
        seed=0
    )
    
    
    
if __name__ == '__main__':
    train_with_DAE()
    
    
    