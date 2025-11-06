#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="mappo"
exp="check"
seed_max=1
ppo_epochs=10
ppo_clip=0.05
steps=10000000
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 200 \
    --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --lr 5e-4 --critic_lr 5e-4 --entropy_coef 0.01 \
    --use_value_active_masks --eval_episodes 32 --save_interval 100000 \
    --use_wandb --use_bilevel
done
