"""Define variables and hyperparameters using argparse"""

import argparse

import torch


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device


def get_args(verbose=True):
    """Call args"""
    parser = argparse.ArgumentParser()

    ### Adjustable parameters

    ### WandB and Logging parameters
    parser.add_argument(
        "--project",
        type=str,
        default="Exp",
        help="WandB project classification",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="log/train_log",
        help="name of the logging folder",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )

    ### Environmental / Running parameters
    parser.add_argument(
        "--env-name",
        type=str,
        default="fourrooms-v0",
        help="This specifies which environment one is working with= FourRooms or CtF1v1, CtF1v2}",
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="HIRO",
        help="In caps",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )

    parser.add_argument(
        "--init-seed",
        type=int,
        default=42,  # 0, 2
        help="seeds for computational stochasticity --seeds 1,3,5,7,9 # without space",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,  # 0, 2
        help="seeds for computational stochasticity --seeds 1,3,5,7,9 # without space",
    )

    ### Algorithmic iterations
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,  # 500
        help="total number of epochs for OC training",
    )

    ### Learning rates
    parser.add_argument(
        "--actor-lr", type=float, default=None, help="Option network lr"
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Option policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )

    ### Algorithmic parameters
    parser.add_argument(
        "--gamma", type=float, default=None, help="discount parameters"
    )
    parser.add_argument(
        "--num-minibatch",
        type=int,
        default=None,
        help="Option policy number of minibatch size for training",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Naive ppo number of minibatch size for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Naive ppo number of minibatch size for training",
    )
    parser.add_argument(
        "--entropy-scaler",
        type=float,
        default=3e-3,
        help="Hierarchical policy entropy scaler",
    )

    ### Resorces
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of threads to use in sampling. If none, sampler will select available threads number with this limit",
    )
    parser.add_argument(
        "--cpu-preserve-rate",
        type=float,
        default=0.95,
        help="For multiple run of experiments, one can set this to restrict the cpu threads the one exp uses for sampling.",
    )

    ### Dimensional params

    parser.add_argument(
        "--actor-fc-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--critic-fc-dim",
        type=list,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )

    # PPO parameters
    parser.add_argument(
        "--frequency", type=int, default=None, help="PPO update per one iter"
    )
    parser.add_argument(
        "--K-epochs", type=int, default=None, help="PPO update per one iter"
    )
    parser.add_argument(
        "--eps-clip",
        type=float,
        default=0.2,
        help="clipping parameter for gradient",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="clipping parameter for gradient",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Used in advantage estimation.",
    )

    # Misc. parameters
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Imports previously trained SF model",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=None,
        help="saves the rendering during evaluation",
    )

    parser.add_argument(
        "--gpu-idx", type=int, default=0, help="gpu idx to train"
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="WandB logging"
    )

    args = parser.parse_args()

    # post args processing
    args.device = select_device(args.gpu_idx, verbose)

    return args
