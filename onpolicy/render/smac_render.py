import torch
import os
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env


def load_actor_critic(actor_path, critic_path):
    actor = torch.load(actor_path, map_location=torch.device('cpu'))
    critic = torch.load(critic_path, map_location=torch.device('cpu'))
    return actor, critic

def create_args():
    from types import SimpleNamespace

    args = SimpleNamespace(
        env_name="StarCraft2",
        map_name="MMM",
        seed=1,
        algorithm_name="mat",
        experiment_name="single",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=True,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        state_pathing_grid=False,
        state_terrain_height=False,
        state_last_action=True,
        state_agent_id=True,
        reward_sparse=False,
        reward_only_positive=True,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="path_to_save_replays/",
        debug=False,
        add_local_obs=True,
        add_move_state=False,
        add_distance_state=False,
        add_enemy_action_state=False,
        add_agent_id=True,
        add_visible_state=False,
        add_xy_state=False,
        use_state_agent = False,
        use_mustalive = False,
        add_center_xy=False,
        use_stacked_frames = False,
    )
    return args

def run_render(actor_path, critic_path, replay_dir="path_to_save_replays/"):
    # Load actor and critic models
    actor, critic = load_actor_critic(actor_path, critic_path)

    # Create environment arguments
    args = create_args()

    # Initialize the StarCraft2 environment
    env = StarCraft2Env(args)

    # Ensure replay directory exists
    os.makedirs(replay_dir, exist_ok=True)

    # Run a single episode for rendering
    obs = env.reset()
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_tensor).squeeze(0).numpy()

        obs, reward, done, info = env.step(action)

    # Save the replay
    env._controller.save_replay(os.path.join(replay_dir, "rendered_replay.SC2Replay"))
    print(f"Replay saved at {os.path.join(replay_dir, 'rendered_replay.SC2Replay')}")

if __name__ == "__main__":
    actor_path = "/work/sdim-lemons/wangchao/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mappo/check/wandb/run-20241211_140438-1qe6objo/files/actor.pt"
    critic_path = "/work/sdim-lemons/wangchao/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mappo/check/wandb/run-20241211_140438-1qe6objo/files/critic.pt"
    replay_dir = "./replays/"

    run_render(actor_path, critic_path, replay_dir)

#/work/sdim-lemons/wangchao/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mappo/check/wandb/run-20241211_140438-1qe6objo/