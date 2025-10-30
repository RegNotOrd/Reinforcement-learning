# test_generalization_fixed.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import argparse
import os

# --- Config ---
MODEL_PATH = "models_dqn/dqn_ckpt_500000_steps.zip"  # adjust path if needed
RESIZE_SHAPE = (48, 48)  # must match training ResizeObservation
SEEDS = [0, 10, 42, 99, 1234]

# --- Discrete action mapping (must match training) ---
steer_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
gas_vals = [0.0, 1.0]
brake_vals = [0.0]
ACTIONS = [np.array([s, g, b], dtype=np.float32)
           for s in steer_vals for g in gas_vals for b in brake_vals]

class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env, actions):
        super().__init__(env)
        self._actions = actions
        self.action_space = gym.spaces.Discrete(len(actions))
    def step(self, action):
        # action is discrete index; convert to continuous triple
        cont = self._actions[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(cont)
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_env_for_eval(render_mode="human"):
    """
    Build an env that:
    - returns resized observations (HxWxC) matching training ResizeObservation
    - can render for human viewing (render_mode)
    """
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    # Apply ResizeObservation so obs returned from reset/step are (H,W,C) == RESIZE_SHAPE
    try:
        # gymnasium
        from gymnasium.wrappers import ResizeObservation
        env = ResizeObservation(env, RESIZE_SHAPE)
    except Exception:
        # older gym
        from gym.wrappers import ResizeObservation
        env = ResizeObservation(env, RESIZE_SHAPE)
    return env

def preprocess_obs_for_model(obs):
    """
    Convert observation from HxWxC (48,48,3) to CxHxW (3,48,48) float32 expected by the model.
    Also handle tuple obs (obs, info) in some gym versions.
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, np.ndarray):
        obs = np.asarray(obs)
    # obs should be HxWxC
    if obs.ndim == 3 and obs.shape[2] == 3:
        # transpose to C,H,W
        return np.transpose(obs, (2, 0, 1)).astype(np.float32)
    # if already channel-first or vector, return as-is
    return obs

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading model:", MODEL_PATH)
    model = DQN.load(MODEL_PATH, device="auto")  # don't pass env here (we'll pass env when stepping)

    # Build an env for stepping and rendering, wrapped for discrete actions
    base_env = make_env_for_eval(render_mode="human")
    env = DiscreteActionWrapper(base_env, ACTIONS)

    for seed in SEEDS:
        print("\n=== Seed", seed, "===")
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < 5000:
            # Convert obs to what the model expects: (3,48,48)
            obs_for_model = preprocess_obs_for_model(obs)

            # Get action (DQN returns a discrete index)
            action, _ = model.predict(obs_for_model, deterministic=True)

            # Step the wrapped env with the discrete index (wrapper will map to continuous)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = bool(terminated or truncated)
            steps += 1

        print(f"Seed {seed}: total reward = {total_reward:.2f}, steps = {steps}")

    env.close()

if __name__ == "__main__":
    main()
