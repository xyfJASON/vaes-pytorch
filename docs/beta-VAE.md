# $\beta$-VAE

> Higgins, Irina, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. "beta-vae: Learning basic visual concepts with a constrained variational framework." In *International conference on learning representations*. 2016.

<br/>



## Introduction

$\beta$-VAE introduces a hyperparameter $\beta$ to balance the reconstruction term and KL term in the loss function, where $\beta=1$ corresponds to the original VAE. Larger $\beta$ creates a trade-off between reconstruction fidelity and the quality of distanglement within the learnt latent representations.

In this repository, $\beta$ can be set by `train.coef_kl` in the configuration file. Training and sampling scripts are the same as the original VAE, see [VAE documentation](./VAE.md) for instructions.

By traversing along one dimension of the latent code and fixing other dimensions, we can find out which dimension controls what kind of semantic.

<br/>



## Visualizations

$\beta=1$ (original VAE):

<p align="center">
  <img src="../assets/vae-celeba-traverse.png" width=70% />
</p>

$\beta=20$:

<p align="center">
  <img src="../assets/vae-beta20-celeba-traverse.png" width=70% />
</p>

