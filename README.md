# GIT

---------------------
**https://github.com/N0Carrot/GIT**

---------------------
## Code
The central file that contains the reconstruction algorithm can be found at ```inversefed/reconstruction_algorithms.py```. The other folders and files are used to define and train the various models and are not central for recovery.

### Setup:
Requirements:
```
pytorch=1.13.1
torchvision=0.14.1
```
You can use [anaconda](https://www.anaconda.com/distribution/) to install our setup by running
```
conda env create -f env.yml
conda activate inv
```
To run ImageNet experiments, you need to download ImageNet and provide its location [or use your own images and skip the ```inversefed.construct_dataloaders``` steps].

