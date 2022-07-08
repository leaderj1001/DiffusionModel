import argparse


def load_args():
    parser = argparse.ArgumentParser("Diffusion Model")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--steps', type=int, default=800000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1.)

    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--step', type=int, default=580000)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--checkpoints', type=str, default=None)

    parser.add_argument('--model_type', type=str, default='ddpm', help='We can choose DDPM, DDIM')
    return parser.parse_args()
