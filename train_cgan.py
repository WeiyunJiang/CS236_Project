#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train script for conditional gan
Created on Wed Nov 10 16:59:22 2021

@author: weiyunjiang
"""

import os
import pprint
import argparse

import torch
import torch.optim as optim

import util
from model import *
from trainer_cgan import Trainer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root_dir, "out"),
        help=(
            "Path to output directory. "
            "A new one will be created if the directory does not exist."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        # required=True,
        default="cGan",
        help=(
            "Name of the current experiment."
            "Checkpoints will be stored in '{out_dir}/{name}/ckpt/'. "
            "Logs will be stored in '{out_dir}/{name}/log/'. "
            "If there are existing checkpoints in '{out_dir}/{name}/ckpt/', "
            "training will resume from the last checkpoint."
        ),
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help=(
            "Resumes training using the last checkpoint in '{out_dir}/{name}/ckpt/' if set. "
            "Throws error if '{out_dir}/{name}/' already exists by default."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for reproducibility."
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=256,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during training.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=150000, help="Number of steps to train for."
    )
    parser.add_argument(
        "--repeat_d",
        type=int,
        default=5,
        help="Number of discriminator updates before a generator update.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Number of steps between model evaluation.",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=5000,
        help="Number of steps between checkpointing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to train on.",
    )
    parser.add_argument(
        "--data_aug",
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False,
        help='data augmentation or not',
    )

    return parser.parse_args()


def train(args):
    r"""
    Configures and trains model.
    """

    # Print command line arguments and architectures
    pprint.pprint(vars(args))

    # Setup dataset
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory 'args.data_dir' is not found.")

    # Check existing experiment
    exp_dir = os.path.join(args.out_dir, args.name)
    if os.path.exists(exp_dir) and not args.resume:
        raise FileExistsError(
            f"Directory '{exp_dir}' already exists. "
            "Set '--resume' if you wish to resume training or "
            "change '--name' if you wish to start a new experiment."
        )

    # Setup output directories
    log_dir = os.path.join(exp_dir, "log")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    for d in [args.out_dir, exp_dir, log_dir, ckpt_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    # Fixed seed
    torch.manual_seed(args.seed)

    # Set parameters
    nz, lr, betas, train_size, eval_size, num_workers = (128, 2e-4, (0.0, 0.9),13000, 1000, 2)

    # Configure models
    
    if args.im_size == 64:
        net_g = cGenerator64_z()
        net_d = cDiscriminator64()
    elif args.im_size == 256:
        net_g = cGenerator256_z()
        net_d = cDiscriminator256()
    else:
        raise NotImplementedError(f"Unsupported image size '{args.im_size}'.")

    # Configure optimizers
    opt_g = optim.Adam(net_g.parameters(), lr, betas)
    opt_d = optim.Adam(net_d.parameters(), lr, betas)

    # Configure schedulers
    sch_g = optim.lr_scheduler.LambdaLR(
        opt_g, lr_lambda=lambda s: 1.0 - ((s * args.repeat_d) / args.max_steps)
    )
    sch_d = optim.lr_scheduler.LambdaLR(
        opt_d, lr_lambda=lambda s: 1.0 - (s / args.max_steps)
    )

    # Configure dataloaders
    train_dataloader, eval_dataloader = util.get_dataloaders_cgan(
        args.data_dir, (args.im_size, args.im_size), 
        args.batch_size, train_size, eval_size, num_workers, args.data_aug,
    )

    # Configure trainer
    trainer = Trainer(
        net_g,
        net_d,
        opt_g,
        opt_d,
        sch_g,
        sch_d,
        train_dataloader,
        eval_dataloader,
        log_dir,
        ckpt_dir,
        torch.device(args.device),
    )

    # Train model
    trainer.train(args.max_steps, args.repeat_d, args.eval_every, args.ckpt_every)


if __name__ == "__main__":
    train(parse_args())
