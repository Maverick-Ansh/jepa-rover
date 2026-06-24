"""
JEPA + MPPI rover navigation (2D continuous) — single-file, runnable.

A 4-wheeled (unicycle-model) rover crosses a 50x50 m hazardous terrain using:
  * a PyTorch JEPA (Joint Embedding Predictive Architecture) world-model that
    *imagines* the future in a learned latent space, and
  * an MPPI (Model Predictive Path Integral) planner that samples many control
    trajectories, rolls them through the JEPA predictor, and cost-weights them
    into one smooth optimal control sequence.

Two spaces to keep in mind
--------------------------
  RAW STATE SPACE (x-space):  physical truth = pose [x, y, theta] + the full,
      partially-unknown terrain risk field.
  JEPA LATENT SPACE (s-space): a compact 32-D vector the ContextEncoder maps a
      local observation into. The Predictor advances it WITHOUT touching the
      environment (s_{t+1} ~= P(s_t, a_t)); planning happens entirely here.

Run:  python rover_2d.py      ->  writes rover_jepa.mp4
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"  # nets are tiny (~50k params); CPU avoids GPU-transfer overhead

# --------------------------------------------------------------------------- #
# 1. CONFIG
# --------------------------------------------------------------------------- #
WORLD = 50.0
V_MAX = 2.5            # max linear speed (m/s)
OMEGA_MAX = 1.5        # max yaw rate (rad/s)
DT = 0.30             # control period (s)
START = np.array([5.0, 5.0, np.deg2rad(40)], dtype=np.float32)
GOAL = np.array([45.0, 45.0], dtype=np.float32)

GRID_N, GRID_SPAN = 7, 6.0          # egocentric sensor grid
OBS_DIM = GRID_N * GRID_N + 1       # 49 risk readings + normalised speed
LAT, ACT = 32, 2                    # latent dim, action dim

TRAIN_ITERS, BATCH, LR, EMA = 1200, 256, 1e-3, 0.99

K, H = 100, 8                       # MPPI samples, horizon
SIGMA = np.array([0.6, 0.7], np.float32)
W_GOAL, W_RISK, W_CTRL, TEMP = 1.0, 3.0, 0.02, 0.18

# --------------------------------------------------------------------------- #
# 2. CONTINUOUS ENVIRONMENT — sum of 2D Gaussian hazard blobs
# --------------------------------------------------------------------------- #
HAZARDS = np.array([
    [15, 20, 4.0, 1.0], [25, 30, 5.0, 1.2], [33, 17, 3.5, 0.9], [38, 36, 4.5, 1.1],
    [20, 40, 3.0, 0.8], [30, 46, 3.2, 0.9], [41, 26, 3.0, 1.0], [12, 34, 3.5, 0.85],
], dtype=np.float32)


def cost_field(x, y):
    """Continuous slip-risk at world coord (x, y); vectorised over arrays."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    c = np.zeros(np.broadcast(x, y).shape, np.float32)
    for cx, cy, s, p in HAZARDS:
        c += p * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * s * s))
    return c


# --------------------------------------------------------------------------- #
# 3. DIFFERENTIAL-DRIVE KINEMATICS (known self-model)
# --------------------------------------------------------------------------- #
def kin_step(state, v, w, dt=DT):
    x, y, th = state
    return np.array([x + v * np.cos(th) * dt,
                     y + v * np.sin(th) * dt,
                     th + w * dt], dtype=np.float32)


def kin_rollout(state, controls):
    """Batched rollout. controls:(K,H,2) -> xs,ys,ths each (K,H+1)."""
    K_, H_, _ = controls.shape
    x = np.full(K_, state[0], np.float32)
    y = np.full(K_, state[1], np.float32)
    th = np.full(K_, state[2], np.float32)
    xs, ys, ths = [x.copy()], [y.copy()], [th.copy()]
    for t in range(H_):
        v, w = controls[:, t, 0], controls[:, t, 1]
        x = x + v * np.cos(th) * DT
        y = y + v * np.sin(th) * DT
        th = th + w * DT
        xs.append(x.copy()); ys.append(y.copy()); ths.append(th.copy())
    return np.stack(xs, 1), np.stack(ys, 1), np.stack(ths, 1)


# --------------------------------------------------------------------------- #
# 4. LOCAL EGOCENTRIC OBSERVATION
# --------------------------------------------------------------------------- #
_o = np.linspace(-GRID_SPAN, GRID_SPAN, GRID_N).astype(np.float32)
_gx, _gy = np.meshgrid(_o, _o)
_gx, _gy = _gx.ravel(), _gy.ravel()


def observe(state, v):
    """Sample the cost field on a body-frame grid (rotated by heading) + speed."""
    x, y, th = state
    c, s = np.cos(th), np.sin(th)
    wx = x + c * _gx - s * _gy
    wy = y + s * _gx + c * _gy
    return np.concatenate([cost_field(wx, wy), [v / V_MAX]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 5. THE THREE JEPA NETWORKS
# --------------------------------------------------------------------------- #
class ContextEncoder(nn.Module):          # (1) f_theta : o_t -> s_t
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, LAT))

    def forward(self, o):
        return self.net(o)


class Predictor(nn.Module):               # (2) P : (s_t, a_t) -> s_{t+1}  (residual)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LAT + ACT, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, LAT))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))   # predict the delta-latent


class TaskEvaluator(nn.Module):           # (3) h : s_t -> expected risk >= 0
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LAT, 64), nn.GELU(),
                                 nn.Linear(64, 1), nn.Softplus())

    def forward(self, z):
        return self.net(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# 6. SELF-SUPERVISED JEPA TRAINING
# --------------------------------------------------------------------------- #
def collect(n_traj=500, T=14):
    """Random-walk the rover; record (o_t, a_t, o_{t+1}, true_risk_t)."""
    O, A, O2, R = [], [], [], []
    for _ in range(n_traj):
        st = np.array([np.random.uniform(3, 47), np.random.uniform(3, 47),
                       np.random.uniform(-np.pi, np.pi)], np.float32)
        v = np.random.uniform(0, V_MAX)
        for _ in range(T):
            w = np.random.uniform(-OMEGA_MAX, OMEGA_MAX)
            v = float(np.clip(v + np.random.uniform(-0.6, 0.6), 0, V_MAX))
            O.append(observe(st, v))
            A.append(np.array([v / V_MAX, w / OMEGA_MAX], np.float32))
            st2 = kin_step(st, v, w)
            st2[0] = np.clip(st2[0], 0, WORLD); st2[1] = np.clip(st2[1], 0, WORLD)
            O2.append(observe(st2, v))
            R.append(float(cost_field(st[0], st[1])))
            st = st2
    t = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=DEV)
    return t(O), t(A), t(O2), t(R)


def var_reg(z):
    """VICReg hinge: keep each latent dim's std >= 1 (prevents collapse)."""
    return F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)).mean()


def train_jepa(enc, pred, evalr, tgt):
    O, A, O2, R = collect()
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()) +
                           list(evalr.parameters()), lr=LR)
    N = O.shape[0]
    for it in range(TRAIN_ITERS):
        idx = torch.randint(0, N, (BATCH,), device=DEV)
        o, a, o2, r = O[idx], A[idx], O2[idx], R[idx]
        s = enc(o)
        with torch.no_grad():
            s2_t = tgt(o2)                      # target latent, stop-gradient
        s2_p = pred(s, a)
        loss = (F.mse_loss(s2_p, s2_t)          # JEPA prediction energy
                + F.mse_loss(evalr(s), r)       # risk grounding
                + var_reg(s) + var_reg(s2_p))   # anti-collapse
        loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():                   # EMA target update
            for pt, ps in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(EMA).add_(ps, alpha=1 - EMA)
        if it % 200 == 0 or it == TRAIN_ITERS - 1:
            print(f"  it {it:4d}  loss={loss.item():.4f}  latent_std={s.std(0).mean():.2f}")
    for m in (enc, pred, evalr):
        m.eval()


# --------------------------------------------------------------------------- #
# 7. MPPI PLANNER — planning inside the JEPA imagination
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mppi(state, v_cur, U, enc, pred, evalr):
    eps = np.random.randn(K, H, 2).astype(np.float32) * SIGMA
    V = U[None] + eps
    V[:, :, 0] = np.clip(V[:, :, 0], 0, V_MAX)
    V[:, :, 1] = np.clip(V[:, :, 1], -OMEGA_MAX, OMEGA_MAX)

    xs, ys, ths = kin_rollout(state, V)                          # analytic body rollout
    goal_cost = np.sqrt((xs[:, -1] - GOAL[0]) ** 2 + (ys[:, -1] - GOAL[1]) ** 2)

    s = enc(torch.tensor(observe(state, v_cur), device=DEV)).repeat(K, 1)
    Vt = torch.tensor(np.stack([V[:, :, 0] / V_MAX, V[:, :, 1] / OMEGA_MAX], -1), device=DEV)
    risk = torch.zeros(K, device=DEV)
    for t in range(H):
        s = pred(s, Vt[:, t, :])                                 # imagine in latent space
        risk = risk + evalr(s)
    risk = risk.cpu().numpy()

    ctrl_cost = (V[:, :, 1] ** 2).sum(1)
    S = W_GOAL * goal_cost + W_RISK * risk + W_CTRL * ctrl_cost
    Sn = (S - S.min()) / (S.max() - S.min() + 1e-9)
    w = np.exp(-Sn / TEMP); w /= w.sum()                         # MPPI softmax weights
    U_new = (w[:, None, None] * V).sum(0)                        # cost-weighted optimal
    return U_new, V, xs, ys, w


# --------------------------------------------------------------------------- #
# 8. CLOSED-LOOP SIMULATION
# --------------------------------------------------------------------------- #
def simulate(enc, pred, evalr, max_steps=140, nviz=60):
    state, v_cur = START.copy(), 0.0
    U = np.zeros((H, 2), np.float32); U[:, 0] = 1.0
    history = []
    for _ in range(max_steps):
        U, V, xs, ys, w = mppi(state, v_cur, U, enc, pred, evalr)
        opt_x, opt_y, _ = kin_rollout(state, U[None])
        keep = np.argsort(w)[-nviz:]
        history.append(dict(state=state.copy(), v=v_cur, sx=xs[keep], sy=ys[keep],
                            sw=w[keep] / w.max(), ox=opt_x[0], oy=opt_y[0],
                            risk=float(cost_field(state[0], state[1]))))
        v_cur, w0 = float(U[0, 0]), float(U[0, 1])
        state = kin_step(state, v_cur, w0)
        state[0] = np.clip(state[0], 0, WORLD); state[1] = np.clip(state[1], 0, WORLD)
        U = np.roll(U, -1, 0); U[-1] = U[-2]
        if np.hypot(state[0] - GOAL[0], state[1] - GOAL[1]) < 1.8:
            break
    return history


# --------------------------------------------------------------------------- #
# 9. ANIMATION
# --------------------------------------------------------------------------- #
def animate(history, out="rover_jepa.mp4"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Polygon, Circle
    from matplotlib.collections import LineCollection

    path = np.array([h["state"] for h in history])
    res = 240
    gx = np.linspace(0, WORLD, res); gy = np.linspace(0, WORLD, res)
    GXf, GYf = np.meshgrid(gx, gy); FIELD = cost_field(GXf, GYf)

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    im = ax.imshow(FIELD, extent=[0, WORLD, 0, WORLD], origin="lower",
                   cmap="inferno", alpha=0.92, zorder=0)
    fig.colorbar(im, fraction=0.046, pad=0.02).set_label("terrain slip-risk")
    ax.contour(GXf, GYf, FIELD, levels=[0.25, 0.5, 0.8], colors="white",
               linewidths=0.4, alpha=0.35)
    ax.plot(*START[:2], "o", color="deepskyblue", ms=11, mec="white", zorder=5, label="start")
    ax.plot(*GOAL, "*", color="lime", ms=22, mec="black", zorder=5, label="goal")
    ax.add_patch(Circle(GOAL, 1.8, fill=False, ec="lime", ls="--", lw=1.2, zorder=4))

    samp = LineCollection([], linewidths=0.7, zorder=2); ax.add_collection(samp)
    opt_line, = ax.plot([], [], color="cyan", lw=3, zorder=4, solid_capstyle="round",
                        label="MPPI optimal plan")
    trail, = ax.plot([], [], color="white", lw=2, alpha=0.9, zorder=3, label="executed path")
    rover = Polygon(np.zeros((3, 2)), closed=True, fc="deepskyblue", ec="white", lw=1.5, zorder=6)
    ax.add_patch(rover)
    hud = ax.text(1.5, 48.5, "", color="white", fontsize=10, va="top",
                  bbox=dict(boxstyle="round", fc="black", alpha=0.55), zorder=7)
    ax.set_xlim(0, WORLD); ax.set_ylim(0, WORLD); ax.set_aspect("equal")
    ax.set_title("JEPA world-model + MPPI rover navigation")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.65)

    def tri(x, y, th, L=1.9, Wd=1.15):
        pts = np.array([[L, 0], [-0.6 * L, Wd], [-0.6 * L, -Wd]], np.float32)
        c, s = np.cos(th), np.sin(th)
        return pts @ np.array([[c, s], [-s, c]], np.float32) + [x, y]

    def update(i):
        h = history[i]
        segs = [np.column_stack([h["sx"][k], h["sy"][k]]) for k in range(len(h["sx"]))]
        samp.set_segments(segs)
        cols = np.zeros((len(segs), 4)); cols[:, 0] = 0.25; cols[:, 1] = 0.9; cols[:, 2] = 1.0
        cols[:, 3] = 0.05 + 0.5 * h["sw"]
        samp.set_color(cols)
        opt_line.set_data(h["ox"], h["oy"])
        trail.set_data(path[:i + 1, 0], path[:i + 1, 1])
        rover.set_xy(tri(h["state"][0], h["state"][1], h["state"][2]))
        d = np.hypot(h["state"][0] - GOAL[0], h["state"][1] - GOAL[1])
        hud.set_text(f"step {i:3d}\nv = {h['v']:.2f} m/s\nrisk = {h['risk']:.2f}\nto goal = {d:4.1f} m")
        return samp, opt_line, trail, rover, hud

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=90, blit=False)
    ani.save(out, writer="ffmpeg", dpi=120, fps=11)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main():
    enc, pred, evalr = ContextEncoder().to(DEV), Predictor().to(DEV), TaskEvaluator().to(DEV)
    tgt = ContextEncoder().to(DEV)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)

    print("training JEPA world-model ...")
    train_jepa(enc, pred, evalr, tgt)
    print("running closed-loop MPPI simulation ...")
    history = simulate(enc, pred, evalr)
    path = np.array([h["state"] for h in history])
    risks = np.array([h["risk"] for h in history])
    reached = np.hypot(path[-1, 0] - GOAL[0], path[-1, 1] - GOAL[1]) < 1.8
    print(f"frames={len(history)} reached_goal={reached} "
          f"path_risk(mean={risks.mean():.3f}, max={risks.max():.3f})")
    out = animate(history)
    print(f"saved {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
