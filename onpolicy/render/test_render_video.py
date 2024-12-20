import torch
import os
import numpy as np
import cv2
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from pyvirtualdisplay import Display

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
        stacked_frames = 1,
    )
    return args

def run_render(actor_path, critic_path, video_path="./videos/game.mp4", fps=20):
    # 创建虚拟显示
    display = Display(visible=0, size=(1400, 900))
    display.start()
    
    try:
        # Load actor and critic models
        actor, critic = load_actor_critic(actor_path, critic_path)

        # Create environment arguments
        args = create_args()
        
        # 设置环境参数以支持渲染
        args.render = True
        
        # Initialize the StarCraft2 environment
        env = StarCraft2Env(args)
        
        # 准备视频写入器
        frames = []
        
        # Run a single episode for rendering
        obs = env.reset()
        done = False
        
        while not done:
            # 获取当前帧的渲染图像
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # 执行动作
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_tensor).squeeze(0).numpy()
            
            obs, reward, done, info = env.step(action)
        
        # 保存为视频
        if len(frames) > 0:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # 获取第一帧的尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # 写入每一帧
            for frame in frames:
                # OpenCV 使用 BGR 格式，需要从 RGB 转换
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # 释放视频写入器
            out.release()
            print(f"Video saved at {video_path}")
        
        env.close()
    
    finally:
        # 确保虚拟显示被关闭
        display.stop()

if __name__ == "__main__":
    actor_path = "/work/sdim-lemons/wangchao/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mappo/check/wandb/run-20241211_140438-1qe6objo/files/actor.pt"
    critic_path = "/work/sdim-lemons/wangchao/onpolicy/onpolicy/scripts/results/StarCraft2/27m_vs_30m/mappo/check/wandb/run-20241211_140438-1qe6objo/files/critic.pt"
    video_path = "./videos/game.mp4"
    
    run_render(actor_path, critic_path, video_path)