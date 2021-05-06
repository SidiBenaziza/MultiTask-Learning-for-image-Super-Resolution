# MultiTask-Learning-for-image-Super-Resolution



## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- tqdm 4.30.0
- tensorboard --logdir=runs



## Prepare dataset

This network was trained on the DIV2K dataset that can be downloaded from here https://data.vision.ee.ethz.ch/cvl/DIV2K/.
We will only look to implement the X2 up-scaling factor for the Super-Resolution module.

These images will need to be run through an encoder-decoder with different QP factors before using them. 
You can find here the necessary executables https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM .

In order to release the burder on the GPU, we have chosen to split the dataset into patches with a pre-defined size in an .npy format . 
Use the prepare_npy_patches.py file for this purpose and change the destination and source directories accordingly.

## Train 

run this command in order to train your network, this requires multiple file path inputs. 

--hr-train-file : The path containing the High Resolution Y Channel npy patches

--hq-train-file : The path containing the Low Resolution (bicubic downsample) Y channel npy patches.

--test-file     : The path containing the degraded Y channel npy patches after running through VVC.

--outputs-dir   : The path where you wish to store the network's weights


  python train_MLT.py --hr-train-file ../../../mnt/DATA/DIV2K_npy/npy/HR_npy \
                      --hq-train-file ../../../mnt/DATA/DIV2K_npy/npy/LR_npy \
                      --test-file ../../../mnt/DATA/DIV2K_npy/npy/LR_qp27_npy \
                      --outputs-dir ../../../mnt/DATA/SRCNN_DATA/outputs_encoded_SRCNN \
                    
## Performance visualization

In order to plot the High-Resolution and Quality-Enhancement performances during training and validation (Loss, PSNR)  we use TensorBoard.

run these commands while being in the repository directory, then follow the guidelines : 

  rm -r runs 
  
  tensorboard --logdir=runs 
  
