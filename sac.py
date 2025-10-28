# train_dqn_local.py
"""
Train DQN on a discretized CarRacing-v3 (pixel observations).
- Reduces memory by resizing frames to 48x48
- Uses buffer_size=100000 to avoid huge RAM allocation
- Saves periodic eval videos to ./dqn_eval_videos
- Logs to TensorBoard (./dqn_tb)
"""
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# === CONFIG ===
ENV_ID = "CarRacing-v3"
MODEL_DIR = "./models_dqn"
VIDEO_DIR = "./dqn_eval_videos"
TB_LOG_DIR = "./dqn_tb"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

# Observation resize to control memory usage
RESIZE_SHAPE = (48, 48)   # (height, width). 48x48 reduces memory a lot.
# Replay buffer size (reduced)
REPLAY_BUFFER_SIZE = 100_000
TOTAL_TIMESTEPS = 500_000  # start small on laptop; increase if you have time

# Discrete action set (compact)
steer_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
gas_vals = [0.0, 1.0]
brake_vals = [0.0]  # keep it simple; add brake if you want
ACTIONS = [np.array([s, g, b], dtype=np.float32) for s in steer_vals for g in gas_vals for b in brake_vals]
print("Using", len(ACTIONS), "discrete actions")

# --- Helper: discrete wrapper mapping discrete index -> continuous action
class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env, actions):
        super().__init__(env)
        self._actions = actions
        self.action_space = gym.spaces.Discrete(len(actions))
    def step(self, action):
        a = self._actions[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(a)
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# --- Environment factories
def make_train_env():
    # training env: render_mode=None (faster), wrapped with ResizeObservation and Monitor
    env = gym.make(ENV_ID, render_mode=None)
    env = Monitor(env)
    # ResizeObservation: gymnasium's wrapper
    try:
        from gymnasium.wrappers import ResizeObservation
        env = ResizeObservation(env, RESIZE_SHAPE)
    except Exception:
        from gym.wrappers import ResizeObservation
        env = ResizeObservation(env, RESIZE_SHAPE)
    # apply discrete wrapper
    env = DiscreteActionWrapper(env, ACTIONS)
    return env

def make_eval_env():
    # eval env: must have render_mode="rgb_array" for VecVideoRecorder
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = Monitor(env)
    try:
        from gymnasium.wrappers import ResizeObservation
        env = ResizeObservation(env, RESIZE_SHAPE)
    except Exception:
        from gym.wrappers import ResizeObservation
        env = ResizeObservation(env, RESIZE_SHAPE)
    env = DiscreteActionWrapper(env, ACTIONS)
    return env

# Create vectorized envs (single env each)
train_env = DummyVecEnv([lambda: make_train_env()])

# Eval env wrapped for video recording.
# VecVideoRecorder requires the inner envs to return rgb_array frames (render_mode="rgb_array")
eval_env = DummyVecEnv([lambda: make_eval_env()])

# record a short video every RECORD_EVERY steps using record_video_trigger
VIDEO_RECORD_EVERY = 50_000  # produce a video at 50k, 100k, ...
VIDEO_LENGTH = 400           # frames per video

def record_trigger(step):
    return (step % VIDEO_RECORD_EVERY) == 0 and step > 0

eval_env = VecVideoRecorder(eval_env,
                            VIDEO_DIR,
                            record_video_trigger=record_trigger,
                            video_length=VIDEO_LENGTH,
                            name_prefix="dqn_eval")

# --- DQN model (tuned to laptop) ---
model = DQN(
    policy="CnnPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=TB_LOG_DIR,
    buffer_size=REPLAY_BUFFER_SIZE,   # critical to reduce memory usage
    learning_starts=5000,
    batch_size=64,
    train_freq=4,   # update every 4 steps (default)
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.02,
    learning_rate=1e-4,
)

# Callbacks: periodic checkpoint saving
checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=MODEL_DIR, name_prefix="dqn_ckpt")

# === Train ===
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback])
finally:
    # Always try to save on interruption
    model.save(os.path.join(MODEL_DIR, "dqn_final"))
    print("Saved final model to", os.path.join(MODEL_DIR, "dqn_final"))
