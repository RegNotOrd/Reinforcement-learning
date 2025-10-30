# eval_dqn_render.py
import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import DQN
import os

MODEL_PATH = "models_dqn/dqn_final.zip"    # adjust if different name
OUT_VIDEO = "dqn_eval_manual.mp4"
ENV_ID = "CarRacing-v3"
RESIZE_SHAPE = (48, 48)   # must match resizing used in training for best performance
ACTIONS = [np.array([s, g, b], dtype=np.float32)
           for s in [-1.0, -0.5, 0.0, 0.5, 1.0]
           for g in [0.0, 1.0]
           for b in [0.0]]

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

# create env with rgb_array so we can capture frames
env = gym.make(ENV_ID, render_mode="rgb_array")
# resize observation if your training used ResizeObservation
try:
    from gymnasium.wrappers import ResizeObservation
    env = ResizeObservation(env, RESIZE_SHAPE)
except Exception:
    from gym.wrappers import ResizeObservation
    env = ResizeObservation(env, RESIZE_SHAPE)

env = DiscreteActionWrapper(env, ACTIONS)

model = DQN.load(MODEL_PATH)

frames = []
obs, _ = env.reset(seed=0)
for _ in range(800):   # 800 frames ≈ 26 seconds at 30FPS
    # DQN expects observations shaped same as during training
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()   # returns rgb_array
    frames.append(frame)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
imageio.mimwrite(OUT_VIDEO, frames, fps=30)
print("Saved eval video:", OUT_VIDEO)
