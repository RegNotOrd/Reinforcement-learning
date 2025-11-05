#!/usr/bin/env python3
"""
car_racing_local.py

Run CarRacing locally and record an MP4 with an HUD that shows steering/gas/brake
and estimates pixel distances from the car to left/right road edges.

Usage:
    python car_racing_local.py --frames 300 --outfile out.mp4
    python car_racing_local.py --frames 300 --outfile out.mp4 --use-rl-model car_model.zip
    python car_racing_local.py --frames 300 --outfile out.mp4 --headless

Notes:
- This script assumes the camera keeps the car near the image center (default CarRacing).
- Distances are in pixels. Converting to meters requires environment/world info.
"""

import argparse
import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio
import platform

# Optional imports; wrapped in try/except so script still runs without them
try:
    import gymnasium as gym
except Exception as e:
    print("ERROR: gymnasium not found. Install with `pip install gymnasium[box2d]`.")
    raise

# pyvirtualdisplay only needed for headless environments
PYVIRTUALDISPLAY_AVAILABLE = True
try:
    from pyvirtualdisplay import Display
except Exception:
    PYVIRTUALDISPLAY_AVAILABLE = False

# stable-baselines3 optional
SB3_AVAILABLE = True
try:
    from stable_baselines3 import PPO
except Exception:
    SB3_AVAILABLE = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CarRacing-v3", help="Gym env name (fallback to v2 if not present).")
    p.add_argument("--frames", type=int, default=300, help="Number of frames to capture.")
    p.add_argument("--fps", type=int, default=30, help="Frames per second for output video.")
    p.add_argument("--outfile", default="car_racing_local_hud.mp4", help="Output mp4 filename.")
    p.add_argument("--use-rl-model", default=None, help="Path to SB3 model (.zip). Optional.")
    p.add_argument("--headless", action="store_true", help="Run headless using xvfb/pyvirtualdisplay (if available).")
    p.add_argument("--show-live", action="store_true", help="Attempt a live preview window (slower).")
    return p.parse_args()

# ----------------- Road detection & HUD drawing -----------------
def detect_road_mask(frame):
    """Heuristic mask for gray road pixels. Input: HxWx3 uint8 numpy array."""
    rgb = frame.astype(np.int16)
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    # gray-ish: RGB components close
    rgb_range = np.maximum(np.maximum(np.abs(R-G), np.abs(R-B)), np.abs(G-B))
    grayish = (rgb_range < 60)
    # avoid grass (green) and very bright areas
    not_green = (G < 150)
    dark_enough = ((R+G+B)/3.0) < 230
    mask = grayish & not_green & dark_enough
    return mask

def compute_left_right_distances(mask, ref_x, scanline_y):
    H, W = mask.shape
    scanline_y = max(0, min(H-1, scanline_y))
    row = mask[scanline_y, :]
    if not row.any():
        return None, None
    # ensure ref_x is on road; if not, move to nearest road pixel
    ref_x = int(max(0, min(W-1, ref_x)))
    if not row[ref_x]:
        indices = np.where(row)[0]
        if indices.size == 0:
            return None, None
        ref_x = int(indices[np.argmin(np.abs(indices - ref_x))])
    # walk left
    lx = ref_x
    while lx >= 0 and row[lx]:
        lx -= 1
    # walk right
    rx = ref_x
    while rx < W and row[rx]:
        rx += 1
    left_edge_x = lx  # -1 means image left boundary
    right_edge_x = rx  # W means image right boundary
    dist_left = ref_x - left_edge_x if left_edge_x >= 0 else ref_x
    dist_right = right_edge_x - ref_x if right_edge_x <= W else W - ref_x
    return dist_left, dist_right

def draw_hud_and_dist(frame, action, dist_left_px, dist_right_px, scanline_y):
    H, W = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    hud_h = int(H * 0.18)
    hud_y = H - hud_h
    draw.rectangle([0, hud_y, W, H], fill=(12,12,12))
    steer, gas, brake = action
    bar_w = int(W * 0.28)
    pad = 20
    sx = pad
    sy = hud_y + 20
    draw.text((sx, sy-16), f"Steer: {steer:+.2f}", fill=(255,255,255))
    draw.rectangle([sx, sy, sx+bar_w, sy+20], fill=(60,60,60))
    pos = int(((steer + 1) / 2.0) * bar_w)
    draw.rectangle([sx+pos-6, sy, sx+pos+6, sy+20], fill=(255,200,0))
    gx = sx + bar_w + 30
    draw.text((gx, sy-16), f"Gas: {gas:.2f}", fill=(255,255,255))
    draw.rectangle([gx, sy, gx+bar_w, sy+20], fill=(60,60,60))
    draw.rectangle([gx, sy+20 - int(gas*20), gx+int(bar_w*gas), sy+20], fill=(50,200,50))
    bx = gx + bar_w + 30
    draw.text((bx, sy-16), f"Brake: {brake:.2f}", fill=(255,255,255))
    draw.rectangle([bx, sy, bx+bar_w, sy+20], fill=(60,60,60))
    draw.rectangle([bx, sy+20 - int(brake*20), bx+int(bar_w*brake), sy+20], fill=(200,50,50))

    cx = W//2
    cy = H//2
    draw.ellipse([cx-6, cy-6, cx+6, cy+6], outline=(255,255,255))
    draw.line([cx, cy-12, cx, cy+12], fill=(255,255,255))
    draw.line([cx-12, cy, cx+12, cy], fill=(255,255,255))

    if dist_left_px is not None and dist_right_px is not None:
        draw.text((pad, hud_y-26), f"Left dist: {dist_left_px:.0f} px   Right dist: {dist_right_px:.0f} px", fill=(255,255,255))
        draw.line([0, scanline_y, W, scanline_y], fill=(180,180,180))
        left_edge_x = max(0, cx - int(dist_left_px))
        right_edge_x = min(W-1, cx + int(dist_right_px))
        draw.line([left_edge_x, scanline_y-10, left_edge_x, scanline_y+10], fill=(255,0,0), width=2)
        draw.line([right_edge_x, scanline_y-10, right_edge_x, scanline_y+10], fill=(255,0,0), width=2)
        draw.line([cx, cy, left_edge_x, scanline_y], fill=(255,100,100), width=2)
        draw.line([cx, cy, right_edge_x, scanline_y], fill=(100,100,255), width=2)
    else:
        draw.text((pad, hud_y-24), "Road edges not detected on scanline", fill=(255,200,0))

    return np.array(img)

# ----------------- Main runner -----------------
def main():
    args = parse_args()

    # headless handling
    display = None
    if args.headless:
        if not PYVIRTUALDISPLAY_AVAILABLE:
            print("ERROR: headless requested but pyvirtualdisplay not installed. Install with `pip install pyvirtualdisplay`.")
            return 1
        # On Linux, try start Xvfb
        try:
            display = Display(visible=0, size=(800,600))
            display.start()
            print("Started virtual display (xvfb).")
        except Exception as e:
            print("Could not start virtual display:", e)
            return 1

    # create env
    env_name = args.env
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except Exception:
        print(f"Falling back from {env_name} to CarRacing-v2...")
        env = gym.make("CarRacing-v2", render_mode="rgb_array")

    obs, _ = env.reset(seed=0)
    frames = []

    # optional RL model
    model = None
    if args.use_rl_model:
        if not SB3_AVAILABLE:
            print("Stable-Baselines3 not installed; will run random actions. Install with `pip install stable-baselines3` if you want to use RL model.")
        else:
            try:
                model = PPO.load(args.use_rl_model)
                print("Loaded RL model:", args.use_rl_model)
            except Exception as e:
                print("Could not load RL model:", e)
                model = None

    # live preview setup (optional)
    live = args.show_live
    if live:
        print("Live preview requested. May be slow on some machines.")

    for i in range(args.frames):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()  # HxWx3 uint8

        # detect road and compute distances
        mask = detect_road_mask(frame)
        H, W = mask.shape
        scanline_y = int(H//2 + H*0.08)
        ref_x = W//2
        dist_left, dist_right = compute_left_right_distances(mask, ref_x, scanline_y)

        frame_with_hud = draw_hud_and_dist(frame, action, dist_left, dist_right, scanline_y)
        frames.append(frame_with_hud)

        if live:
            try:
                # simple quick preview using PIL.Image.show (opens external viewer repeatedly)
                # For a true windowed preview, user can integrate OpenCV (not added by default).
                Image.fromarray(frame_with_hud).show()
            except Exception:
                pass

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    if display is not None:
        display.stop()

    # write video
    try:
        imageio.mimwrite(args.outfile, frames, fps=args.fps, macro_block_size=None)
        print("Saved:", args.outfile)
    except Exception as e:
        print("Failed to write video:", e)
        return 1

    # optionally attempt to open file using OS default player
    try:
        if platform.system() == "Darwin":
            os.system(f"open {args.outfile}")
        elif platform.system() == "Windows":
            os.startfile(args.outfile)  # type: ignore
        else:
            # Linux
            os.system(f"xdg-open {args.outfile} >/dev/null 2>&1 &")
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
