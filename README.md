# WAE-WGAN

This repository contians an reimplmentation of [WAE](https://arxiv.org/abs/1711.01558) with Tensorflow.

I made one tweak on top of the paper. I used Wasserstein distance to penalize an encoder Q_\phi.
In order to do that, I trained Discriminator D_\gamma through critic loss with gradient penalty as Gularjani etal. suggested in [[improved WGAN]](https://arxiv.org/abs/1704.00028).

I (personally) believe that this implementation is much clearer and easy to read (, and more importantly, the code almost exactly matches with the algorithm shows on the paper), so I hope it will help someone who wants to digin more! Enjoy :beer:!

## Requirements

- Python 3.x (tested with Python 3.5)
- TF v1.x (tested with 1.7rc0)
- tqdm
- and etc... (please report if you find other deps.)

## Results

- Trained with GTX-1080 Ti GPU for abouth 30 minutes
- Learning statistics
  ![learning_stat](/assets/learning_stat.png)
- Reconstruction Results
  ![recon](/assets/recon.png)
  (top): original images from MNIST validation set, (bottom): reconstructed image

  It seems not sharp as the authors suggest, but it might due to not enough training and untuned hyperparameters such as lambda, number of layers, or etc.
- Random Sampled Images
  ![random_sample](/assets/random_sample.png)


## TODO

- [ ] Other datasets (CIFAR-10, CelebA)
- [ ] WAE-GAN
- [ ] WAE-MMD

## Acknowledgement

- The author's original TF implementation [link](https://github.com/tolstikhin/wae)
