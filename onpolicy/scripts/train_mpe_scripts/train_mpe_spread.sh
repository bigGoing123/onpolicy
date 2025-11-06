#!/bin/sh
env="MPE"
scenario="simple_spread" 
num_landmarks=10
num_agents=10
algo="mat" 
exp="single"
seed_max=1
prefix_name="10v10"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 16 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --n_block 1 --n_embd 64  --use_eval  --use_bilevel --n_eval_rollout_threads 50 --eval_episodes 20 --prefix_name ${prefix_name} \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4  --clip_param 0.05 --save_interval 100  \
    --use_wandb
done
