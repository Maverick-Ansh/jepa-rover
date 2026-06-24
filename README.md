# 🤖 JEPA-Rover — world-model navigation with MPPI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/jepa-rover/blob/master/notebooks/jepa_rover_2d.ipynb)

A rover crosses a hazardous, continuous terrain by **imagining the future in a
learned latent space** (a PyTorch **JEPA** — Joint Embedding Predictive
Architecture) and planning smooth controls with **MPPI** (Model Predictive Path
Integral). Everything is built from scratch and heavily instrumented so you can
inspect every internal: the sensed terrain patch, the raw latent vector, the
predictor's prediction error, the evaluator's risk calibration, and the MPPI
candidate costs/weights.

## The core idea: two spaces

| | **Raw state space** (`x`) | **JEPA latent space** (`s`) |
|---|---|---|
| What | pose `[x,y,θ]` + the *unknown* terrain risk field | a compact `ℝ³²` vector |
| Dynamics | true physics / sensors | a neural net `s_{t+1} ≈ P(s_t, a_t)` |
| Used for | the real executed step + rendering | **planning / imagination** |

JEPA never reconstructs the terrain. It only learns to **predict the next latent
embedding** of the world given an action, with the prediction error measured in
`s`-space — so the model keeps only what matters for the task (risk) and throws
away the rest.

## Architecture

1. **ContextEncoder** `f_θ: o_t → s_t` — compresses the 50-D local observation
   (a 7×7 egocentric risk patch + speed) into a 32-D latent.
2. **Predictor** `P: (s_t, a_t) → ŝ_{t+1}` — *residual* world-model that advances
   the latent **without touching the environment**. MPPI rolls this 8 steps ×
   100 trajectories in a single batched tensor.
3. **TaskEvaluator** `h: s_t → risk ≥ 0` — reads a scalar danger score from a latent.

Trained self-supervised with the JEPA energy
`‖P(f_θ(o_t),a_t) − sg[f_ξ(o_{t+1})]‖²` against an **EMA target encoder** `f_ξ`,
plus a **VICReg variance term** (anti-collapse) and risk grounding.

The rover's **own kinematics are analytic** (it knows how its wheels move it);
JEPA is used only for the genuinely unknown part — *how risky is the terrain I'm
driving into, several steps ahead?*

## Results (2D, `rover_2d.py`)

- Reaches the goal (final distance < 1.8 m tolerance).
- Path-risk **max 0.80 / mean 0.25** vs **1.08** for the naive straight line.
- World-model beats a "predict no change" baseline ~**7×**; risk calibration
  correlation **0.998**; latent std ≈ 2.9 (no collapse).

## Run

```bash
pip install -r requirements.txt
python rover_2d.py          # trains, simulates, writes rover_jepa.mp4
```

Runs on CPU in ~1 minute (the networks are ~50k params total).

## Tunable knobs (top of `rover_2d.py`)

| knob | effect |
|---|---|
| `W_RISK` | higher → wider/safer detours; lower → straighter/riskier |
| `K`, `H`, `SIGMA`, `TEMP` | planner sample count, horizon, exploration, greediness |
| `LAT`, `TRAIN_ITERS`, `EMA` | world-model capacity & training stability |

## Roadmap

- [x] 2D continuous terrain, unicycle kinematics, JEPA + MPPI
- [ ] **3D version**: elevation terrain, slope-aware traction/slip dynamics,
      noisy partial observations, real-world-oriented cost (slope, roughness,
      hard obstacles), 3D visualization

## License

MIT
