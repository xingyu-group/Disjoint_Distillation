## Disjoint-Distillation
combination of the generative and discriminative methods


### Working progress
#### Program completetion
- [x] Computation of ideal dice
- [x] Computation of Average Precision


#### Performance
- [x] Denoising augmentation + Disjoint_distillation on demo dataset
  - Epoch:19, Pixel Auroc:0.850, Sample Auroc:0.794, Average Precision:0.030, DICE:0.086
  - The demo dataset doesn't contain enough training sample of the model to optimize
- [x] Denoising augmentation + Disjoint_distillation on the train/test dataset split by me
  - Epoch:59, Pixel Auroc:0.984, Sample Auroc:0.912, Average Precision:0.759, DICE:0.274
  - The performance boost only by chaning the dataset quantity, which proves the number of training sample is essential for the model
  - Also, the AP is 6% lower that DAE, but DICE is a lot lower.
 

#### TODOs:
- [ ] Test on the brain denoising baseline. see how the detection results are like
- [ ] Test the binary mask: soft label & hard label comparison
- [ ] Try training the model with different augmentation techniques, see if it can boost the performance
