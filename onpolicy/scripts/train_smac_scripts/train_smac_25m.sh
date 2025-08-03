#!/bin/sh
env="StarCraft2"
map="25m"
algo="happo"
exp="check"
seed_max=3
ppo_epochs=15
ppo_clip=0.05
steps=2000000
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 8 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps ${steps} --ppo_epoch 10 --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip}  \
    --use_value_active_masks  --eval_episodes 32 --save_interval 100000 --use_bilevel --use_wandb
done
