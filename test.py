import time
import gymnasium as gym
import gymnasium_robotics
import numpy as np

# 依照您的指示註冊環境
# 注意：新版 gymnasium_robotics 通常 import 後會自動註冊，但這裡強制執行以符合您的需求
try:
    gym.register_envs(gymnasium_robotics)
except Exception:
    pass # 如果已經註冊過，忽略錯誤

def run_sequential(env_id, total_steps):
    print(f"--- 開始測試 Sequential (單一環境) ---")
    env = gym.make(env_id)
    
    # 先 Reset 一次拿到初始狀態
    env.reset()
    action_space = env.action_space
    
    start_time = time.perf_counter()
    
    for _ in range(total_steps):
        action = action_space.sample()
        # 一般環境的 step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            env.reset()
            
    end_time = time.perf_counter()
    env.close()
    
    duration = end_time - start_time
    fps = total_steps / duration
    print(f"Sequential 完成! 耗時: {duration:.4f}s, FPS: {fps:.2f}\n")
    return fps

def run_async_vector(env_id, total_steps, num_envs):
    print(f"--- 開始測試 AsyncVectorEnv (並行數: {num_envs}) ---")
    
    # 確保每個環境跑的步數總和等於 total_steps
    # 例如總共要跑 10000 步，有 4 個環境，則每個環境跑 2500 步
    steps_per_env = total_steps // num_envs
    
    # 建立異步向量環境
    # context="spawn" 通常在 Windows/Mac 是必須的，Linux 可以用 "fork" (預設)
    # 這裡讓 gym 自動決定
    envs = gym.vector.AsyncVectorEnv(
        [lambda: gym.make(env_id) for _ in range(num_envs)]
    )
    
    envs.reset()
    action_space = envs.action_space
    
    start_time = time.perf_counter()
    
    # 注意：這裡的迴圈次數變少了，因為一次推進 num_envs 步
    for _ in range(steps_per_env):
        # Sample 出來的 action shape 是 (num_envs, action_dim)
        actions = action_space.sample()
        
        # VectorEnv 會自動處理 Reset (Auto-reset)，不用像上面那樣手動寫 if done
        obs, rewards, terms, truncs, infos = envs.step(actions)
        
    end_time = time.perf_counter()
    envs.close()
    
    duration = end_time - start_time
    fps = total_steps / duration # 總步數 / 總時間
    print(f"AsyncVectorEnv 完成! 耗時: {duration:.4f}s, FPS: {fps:.2f}\n")
    return fps

if __name__ == "__main__":
    # 參數設定
    ENV_ID = "FetchPush-v4"
    TOTAL_STEPS = 100000  # 總共要採樣多少步資料
    NUM_ENVS = 15         # 您想測試的並行環境數量 (建議設為 CPU 核心數)

    print(f"測試環境: {ENV_ID}")
    print(f"總採樣步數: {TOTAL_STEPS}")
    print("========================================\n")

    # 1. 測試單環境速度
    seq_fps = run_sequential(ENV_ID, TOTAL_STEPS)

    # 2. 測試多進程環境速度
    async_fps = run_async_vector(ENV_ID, TOTAL_STEPS, NUM_ENVS)

    # 總結
    print("================ 結果分析 ================")
    print(f"單線程 FPS: {seq_fps:.2f}")
    print(f"多進程 FPS: {async_fps:.2f} (x{NUM_ENVS} Envs)")
    
    ratio = async_fps / seq_fps
    if ratio > 1:
        print(f"結論: AsyncVectorEnv 比單線程快 {ratio:.2f} 倍")
    else:
        print(f"結論: AsyncVectorEnv 比單線程慢 (只有單線程的 {ratio:.2f} 倍)")
        print("原因可能是環境計算量太小，導致 Python 多進程通訊開銷大於計算收益。")