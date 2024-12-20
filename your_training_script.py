from pysc2.lib import run_configs
import portpicker
import time

def setup_sc2_env():
    max_retry = 3
    retry_count = 0
    
    while retry_count < max_retry:
        try:
            # 尝试找到可用端口
            port = portpicker.pick_unused_port()
            
            run_config = run_configs.get()
            with run_config.start(game_ports=[port]) as controller:
                # 你的其他代码...
                return controller
                
        except Exception as e:
            print(f"尝试 {retry_count + 1}/{max_retry} 失败: {e}")
            retry_count += 1
            time.sleep(5)  # 等待5秒后重试
            
    raise Exception("无法启动SC2环境，请检查进程和端口") 