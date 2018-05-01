# WAE-WGAN

This repository contians an reimplmentation of [WAE](https://arxiv.org/abs/1711.01558) with Tensorflow.

I made one tweak on top of the paper. I used Wasserstein distance to penalize an encoder $Q_\phi$.
In order to do that, I trained Discriminator $D_\gamma$ through critic loss with gradient penalty as Gularjani etal. suggested in [[improved WGAN]](https://arxiv.org/abs/1704.00028).

I (personally) believe that this implementation is much clearer and easy to read (, and more importantly, the code almost exactly matches with the algorithm shows on the paper), so I hope it will help someone who wants to digin more! Enjoy :beer:!

## (Updates: 2018-May-1)

- MMD is implemented
    - Question 1: What is inverse multiscale kernel? The formula looks a little bit different from other resources..
    - Qeustion 2: On its original implementation, why MMD is evaluated on multiple scale (differnt C values) even though true scale is given as a prior? Doesn't it result in wrong MMD values and make Q_z diverge from P_z?

## Requirements

- Python 3.x (tested with Python 3.5)
- TF v1.x (tested with 1.7rc0)
- tqdm
- and etc... (please report if you find other deps.)

## Run

### Training

```
python main.py
```

Check main.py file to change target dataset or to adjust hyperparmeters such as z_dim, and etc...

### Inference

See the `MNIST Plot.ipynb` and `CelebA Plot.ipynb` with Jupyter Notebook.

A pretrained model for both MNIST is included on the repository while a model for CelebA is uploaded on [this place](https://utexas.box.com/s/pmgpb78aeha2bvh9cbl8euzth9e2u449).
Please download the zip file and decompress it on `assets/pretrained_models/celeba/last*`. Or, you can easily modify a path at the first cell on the notebook.

## Results

### MNIST

- Trained with GTX-1080 Ti GPU for about 30 minutes
- Learning statistics
  ![learning_stat](/assets/learning_stat.png)
- Reconstruction Results
  ![recon](/assets/recon.png)
  (top): original images from MNIST validation set, (bottom): reconstructed image

  It seems not sharp as the authors suggest, but it might due to not enough training and untuned hyperparameters such as lambda, number of layers, or etc.
- Random Sampled Images
  ![random_sample](/assets/random_sample.png)


### CelebA

- Trained with GTX-1080 Ti GPU for about 1 day.
- Encode and Decode images size of 64 by 64.
- Reconstruction Results
  ![celeba_recon](/assets/celeba_recon.png)
  (top): original images from CelebA validation set, (bottom): reconstructed image
- Random Sampled Images
  ![celeba_random_sample](/assets/celeba_sample.png)
  With fully trained model, the results seem pretty nice! Can we still say that AE-variants generating blurry images?
- Intepolation on $z$ space.
  ![celeba_linear_interoplation](/assets/celeba_interpol.png)


## TODO

- [x] Other datasets (CelebA, or CIFAR10)
- [x] WAE-MMD
- [ ] WAE-GAN

## Acknowledgement

- The author's original TF implementation [link](https://github.com/tolstikhin/wae)
