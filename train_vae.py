import os
import math
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.logger import StatusTracker, get_logger
from utils.misc import get_time_str, check_freq, get_data_generator
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-r', '--resume', type=str, default=None,
        help='Path to a checkpoint directory, or `best`, or `latest`',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def main():
    # ARGS & CONF
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            conf_yaml=OmegaConf.to_yaml(conf),
            exist_ok=args.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=conf.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    if conf.train.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f'Batch size should be divisible by number of processes, '
            f'get {conf.train.batch_size} % {accelerator.num_processes} != 0'
        )
    batch_size_per_process = conf.train.batch_size // accelerator.num_processes
    train_set = instantiate_from_config(conf.data)
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size_per_process,
        shuffle=True, drop_last=True, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODELS
    encoder: nn.Module = instantiate_from_config(conf.encoder)
    decoder: nn.Module = instantiate_from_config(conf.decoder)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'#params. of encoder: {sum(p.numel() for p in encoder.parameters())}')
    logger.info(f'#params. of decoder: {sum(p.numel() for p in decoder.parameters())}')
    logger.info('=' * 50)

    # BUILD OPTIMIZERS
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer: torch.optim.Optimizer = instantiate_from_config(conf.train.optim, params=parameters)

    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
        # load models
        ckpt = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        logger.info(f'Successfully load models from {ckpt_path}')
        # load optimizer
        ckpt = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save models
        accelerator.save(dict(
            encoder=accelerator.unwrap_model(encoder).state_dict(),
            decoder=accelerator.unwrap_model(decoder).state_dict(),
        ), os.path.join(save_path, 'model.pt'))
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    encoder, decoder, optimizer, train_loader = (
        accelerator.prepare(encoder, decoder, optimizer, train_loader)  # type: ignore
    )

    accelerator.wait_for_everyone()

    def run_step(x):
        optimizer.zero_grad()
        x = x[0] if isinstance(x, (tuple, list)) else x
        scale = x.shape[1] * x.shape[2] * x.shape[3]
        mean, logvar = encoder(x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        recx = decoder(z)
        loss_rec = F.mse_loss(recx, x)
        loss_kl = torch.mean(torch.sum(mean ** 2 + torch.exp(logvar) - logvar - 1, dim=1))
        loss_kl = loss_kl / scale
        loss = loss_rec + conf.train.coef_kl * loss_kl
        accelerator.backward(loss)
        optimizer.step()
        return dict(
            loss_rec=loss_rec.item(),
            loss_kl=loss_kl.item(),
            lr=optimizer.param_groups[0]['lr'],
        )

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        sample_z = torch.randn((conf.train.n_samples, conf.encoder.params.z_dim), device=device)
        sample_x = accelerator.unwrap_model(decoder)(sample_z)
        save_image(
            sample_x, savepath,
            nrow=math.ceil(math.sqrt(conf.train.n_samples)),
            normalize=True, value_range=(-1, 1),
        )

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        tqdm_kwargs=dict(
            desc='Epoch',
            leave=False,
            disable=not accelerator.is_main_process,
        )
    )
    while step < conf.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        encoder.train()
        decoder.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        encoder.eval()
        decoder.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


if __name__ == '__main__':
    main()
