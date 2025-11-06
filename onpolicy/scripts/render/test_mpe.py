#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../../")
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import os
import sys
import time
from pathlib import Path
import argparse
import numpy as np
import imageio


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of players")
    parser.add_argument('--num_good_agents', type=int,
                        default=1, help="number of good agents")
    parser.add_argument('--num_adversaries', type=int,
                        default=2, help="number of adversaries")
    parser.add_argument('--gif_dir', type=str,
                        default=None, help="gif save directory")
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")
    device='cpu'

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    gif_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/renders") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not gif_dir.exists():
        os.makedirs(str(gif_dir))
    all_args.gif_dir = gif_dir


    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
        # env init
    envs = make_eval_env(all_args)
    eval_envs = make_eval_env(all_args)
    num_agents = all_args.num_agents
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    runner = Runner(config)

    print("Rendering...")
    runner.render()

    print("Finished.")
    
    # post process
    envs.close()



if __name__ == "__main__":
    main(sys.argv[1:])
