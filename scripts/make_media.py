"""Render compact GIFs of the 2D and 3D runs for the README. -> assets/*.gif"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)            # so assets/ and module imports resolve from repo root
sys.path.insert(0, ROOT)

import numpy as np
import torch

os.makedirs("assets", exist_ok=True)

# ---- 2D ----
import rover_2d as r2
torch.manual_seed(0); np.random.seed(0)
enc = r2.ContextEncoder(); pred = r2.Predictor(); evalr = r2.TaskEvaluator()
tgt = r2.ContextEncoder(); tgt.load_state_dict(enc.state_dict())
for p in tgt.parameters(): p.requires_grad_(False)
print("2D: training ..."); r2.train_jepa(enc, pred, evalr, tgt)
hist = r2.simulate(enc, pred, evalr)
print("2D: rendering gif ...")
r2.animate(hist, "assets/demo_2d.gif", dpi=80, fps=12, stride=2)
print("  ->", os.path.getsize("assets/demo_2d.gif") / 1e6, "MB")

# ---- 3D ----
import rover_3d as r3
torch.manual_seed(1); np.random.seed(1)
enc3 = r3.Enc3(); pred3 = r3.Pred3(); evalr3 = r3.Eval3()
tgt3 = r3.Enc3(); tgt3.load_state_dict(enc3.state_dict())
for p in tgt3.parameters(): p.requires_grad_(False)
print("3D: training ..."); r3.train_jepa(enc3, pred3, evalr3, tgt3)
hist3, reached = r3.simulate(enc3, pred3, evalr3)
print("3D: reached =", reached, "rendering gif ...")
r3.animate(hist3, "assets/demo_3d.gif", dpi=62, fps=12, stride=3)
print("  ->", os.path.getsize("assets/demo_3d.gif") / 1e6, "MB")
