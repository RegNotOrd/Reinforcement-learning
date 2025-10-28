# agent_fix.py
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

MODEL_PATH = "models_dqn/dqn_ckpt_500000_steps.zip"   # adjust if needed
ENV_ID = "CarRacing-v3"
RESIZE_SHAPE = (48, 48)  # must match training ResizeObservation
# discrete actions used in training (same as before)
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

# Build env that returns observations resized the same way as training
env = gym.make(ENV_ID, render_mode="human")  # show a window
# apply ResizeObservation so obs returned by reset/step match training obs shape
try:
    from gymnasium.wrappers import ResizeObservation
    env = ResizeObservation(env, RESIZE_SHAPE)
except Exception:
    from gym.wrappers import ResizeObservation
    env = ResizeObservation(env, RESIZE_SHAPE)

env = DiscreteActionWrapper(env, ACTIONS)

model = DQN.load(MODEL_PATH)

obs, _ = env.reset(seed=0)
try:
    for _ in range(2000):
        # obs is HxWxC (48,48,3) after ResizeObservation; convert to CHW to match trained model
        if isinstance(obs, tuple):
            # some gym versions return (obs, info)
            obs = obs[0]
        obs_chw = np.transpose(obs, (2, 0, 1))  # (3,48,48)
        action, _ = model.predict(obs_chw, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        time.sleep(0.02)
finally:
    env.close()
