#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference simple_spread
num_landmarks=3
num_agents=3
algo="mat"
exp="single"
seed=1

# xvfb-run -s "-screen 0 1400x900x24" bash test_mpe.sh
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python render_mpe.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks \
 ${num_landmarks} --seed ${seed} --n_rollout_threads 1 --save_gifs --episode_length 100 \
 --n_training_threads 1 --n_rollout_threads 1 --use_render  --render_episodes 2 --ifi 0.1 \
 --model_dir  /work/sdim-lemons/wangchao/onpolicy/onpolicy/scripts/results/MPE/simple_spread/mat/check/wandb/latest-run/files/transformer_2400.pt \