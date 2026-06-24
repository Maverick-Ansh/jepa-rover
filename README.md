# 🤖 JEPA-Rover — world-model navigation with MPPI

**2D:** [![Open 2D In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/jepa-rover/blob/master/notebooks/jepa_rover_2d.ipynb) &nbsp; **3D / real-world:** [![Open 3D In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/jepa-rover/blob/master/notebooks/jepa_rover_3d.ipynb)

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
python rover_2d.py          # 2D: trains, simulates, writes rover_jepa.mp4
python rover_3d.py          # 3D: trains on noisy data, simulates, writes rover3d.mp4
```

Both run on CPU in ~1–2 minutes (the networks are tiny).

## The 3D / real-world version (`rover_3d.py`)

Lifts the demo onto **2.5-D elevation terrain** and adds the things that make
field robotics hard:

- **Elevation map** `z = H(x,y)` (hills, a crater, a steep central massif),
  analytic so slope is exact.
- **Traversability cost** fuses **slope** (rollover/slip) and **roughness/rocks**.
- **Stochastic slope-aware dynamics**: traction drops on steep ground, gravity
  causes **downhill slip**, plus **process + heading + sensor noise**. The planner
  uses only a *nominal* model — the model error is corrected by replanning, as on
  real hardware.
- **Noisy, partial observation**: a multi-channel egocentric patch (relative
  elevation + traversal risk) + IMU-like pitch/roll.
- **Safety fusion**: learned **JEPA risk** + a **hard analytic rollover constraint**
  (`slope < MAX_SLOPE` from the onboard elevation map).
- **3D animation**: risk-shaded surface with the hallucinated MPPI fan and chosen
  plan draped on the terrain, plus a top-down risk map.

**Verified results:** reaches the goal (final dist 1.87 m) under stochastic
dynamics, slope max **0.61 < 0.85** rollover limit, **0 rollover breaches**;
risk corr **0.998**, predictor beats no-op **14×**.

## Tunable knobs (top of `rover_2d.py`)

| knob | effect |
|---|---|
| `W_RISK` | higher → wider/safer detours; lower → straighter/riskier |
| `K`, `H`, `SIGMA`, `TEMP` | planner sample count, horizon, exploration, greediness |
| `LAT`, `TRAIN_ITERS`, `EMA` | world-model capacity & training stability |

## Roadmap

- [x] 2D continuous terrain, unicycle kinematics, JEPA + MPPI
- [x] **3D version**: elevation terrain, slope-aware traction/slip dynamics,
      noisy partial observations, real-world cost (slope + roughness), hard
      rollover-safety constraint, 3D visualization
- [ ] learned (vs analytic) self-model; multi-goal missions; real elevation data (DEM/GeoTIFF)

## License

MIT
