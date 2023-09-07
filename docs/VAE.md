# VAE

> Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." *arXiv preprint arXiv:1312.6114* (2013).

<br/>



## Training

```shell
accelerate-launch train.py [-c CONFIG] [-e EXP_DIR] [--xxx.yyy zzz ...]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Results (logs, checkpoints, tensorboard, etc.) of each run will be saved to `EXP_DIR`. If `EXP_DIR` is not specified, they will be saved to `runs/exp-{current time}/`.
- To modify some configuration items without creating a new configuration file, you can pass `--key value` pairs to the script.

For example, to train on CelebA with default settings:

```shell
accelerate-launch train.py -c ./configs/vae-celeba.yaml
```

<br/>



## Sampling

```shell
python sample.py -c CONFIG \
                 [--seed SEED] \
                 [--mode {sample,interpolate}] \
                 --weights WEIGHTS \
                 --n_samples N_SAMPLES \
                 --save_dir SAVE_DIR \
                 [--batch_size BATCH_SIZE] \
                 [--n_interpolate N_INTERPOLATE]
```

- Choose a sampling mode by `--mode MODE`, the options are:
  - `sample` (default): randomly sample images
  - `interpolate`: sample two random images and interpolate between them. Use `--n_interpolate` to specify the number of images in between.

<br/>



## Results



### CelebA (64x64)

<p align="center">
  <img src="../assets/vae-celeba.png" width=30% />
</p>
<p align="center">
  <img src="../assets/vae-celeba-interpolate.png" width=60% />
</p>
