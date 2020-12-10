# GANs
Several GAN implementations

## Files

##### cDCGAN.py

Implementation of cDCGAN with Pytorch.

##### DP-cGAN.py

Implementation of [DP-cGAN](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Torkzadehmahani_DP-CGAN_Differentially_Private_Synthetic_Data_and_Label_Generation_CVPRW_2019_paper.pdf) with Pytorch.

##### DP-cDCGAN.py

Implementation of DP-cDCGAN with Pytorch.

##### attempts(Directory)

Including several problematic implementations, may not perform well.

- cDCGAN-GP.py

  Implementation of cDCGAN with Pytorch and using the wGAN gradient penalty training strategy.

- DP-cDCGAN-GP.py

  Implementation of DP-cDCGAN with Pytorch and using the wGAN gradient penalty training strategy.

- DP-cDCGAN_tf2.py

  Implementation of DP-cDCGAN with Tensorflow 2.x.

- DP-cGAN_tf2.py

  Implementation of DP-cGAN with Tensorflow 2.x.

##### conf(Directory)

Use various json files to specify the type of training to be performed.

## How to run

Run the DP-cDCGAN for instance:

```bash
python DP-cDCGAN.py -conf conf/mnist_dpcDCGAN.json
```