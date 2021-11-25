import os
import pprint
import argparse

from tqdm import tqdm
import torch
from torchmetrics.image.fid import NoTrainInceptionV3
import torch.utils.tensorboard as tbx

import util
from model import *
from trainer_cgan import evaluate, prepare_data_for_cgan, prepare_data_for_inception

def test_log(self, metrics, samples, logger):
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.items():
            logger.add_scalar(k, v, 1)
        logger.add_image("Samples", samples, 1)
        logger.flush()
        
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
        "--ckpt_path",
        type=str,
        default = "out/cgan_z_256_1116/ckpt/110000.pth",
        # required=True,
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        required=True,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Generate Inception embeddings used for leaderboard submission.",
    )

    return parser.parse_args()


def generate_submission(net_g, dataloader, nz, device, path="submission.pth"):
    r"""
    Generates Inception embeddings for leaderboard submission.
    """

    net_g.to(device).eval()
    inception = NoTrainInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device)

    with torch.no_grad():
        real_embs, fake_embs = [], []
        for data, _ in tqdm(dataloader, desc="Generating Submission"):
            reals, z = prepare_data_for_gan(data, nz, device)
            fakes = net_g(z)
            reals = inception(prepare_data_for_inception(reals, device))
            fakes = inception(prepare_data_for_inception(fakes, device))
            real_embs.append(reals)
            fake_embs.append(fakes)
        real_embs = torch.cat(real_embs)
        fake_embs = torch.cat(fake_embs)
        embs = torch.stack((real_embs, fake_embs)).permute(1, 0, 2).cpu()

    torch.save(embs, path)


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """
    folder_path =  os.path.dirname(args.out_dir, args.name)
    log_path = os.path.join(folder_path, "log_test")
    for d in [log_path]:
        if not os.path.exists(d):
            os.mkdir(d)
    logger = tbx.SummaryWriter(log_path)
    # Set parameters
    nz, eval_size, num_workers = (
        128,
        4000 if args.submit else 10000,
        4,
    )

    # Configure models
    if args.im_size == 32:
        net_g = Generator32()
        net_d = Discriminator32()
    elif args.im_size == 64:
        net_g = Generator64()
        net_d = Discriminator64()
    elif args.im_size == 256:
        net_g = cGenerator256_z()
        net_d = cDiscriminator256()
    else:
        raise NotImplementedError(f"Unsupported image size '{args.im_size}'.")

    # Loads checkpoint
    state_dict = torch.load(args.ckpt_path)
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, _, test_dataloader = util.get_dataloaders_cgan(
        args.data_dir, (args.im_size, args.imsize), args.batch_size, eval_size, num_workers, data_aug=False,
    )

    if args.submit:
        # Generate leaderboard submission
        generate_submission(net_g, test_dataloader, nz, args.device)

    else:
        # Evaluate models
        metrics, sample = evaluate(net_g, net_d, test_dataloader, args.device)
        pprint.pprint(metrics)
        test_log(metrics, sample, logger)

if __name__ == "__main__":
    eval(parse_args())
