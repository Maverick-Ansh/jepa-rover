"""
JEPA + MPPI rover navigation (3D / real-world oriented) — single-file, runnable.

Lifts the 2D demo onto 2.5-D elevation terrain and makes the world much closer to
a real field deployment:

  * 2.5-D elevation map  z = H(x,y)  (hills, a crater, a steep central massif),
    analytic so slope is exact.
  * Traversability cost fuses real hazards: slope (rollover/slip) + roughness/rocks.
  * Stochastic slope-aware dynamics: traction drops on steep ground, gravity causes
    downhill slip, plus process + heading noise. The planner uses only a *nominal*
    model; the resulting model error is corrected by receding-horizon replanning.
  * Noisy, partial observation: a multi-channel egocentric patch (relative elevation
    + traversal risk) plus IMU-like pitch/roll, all with sensor noise.
  * Safety fusion: the learned JEPA risk is combined with a hard analytic rollover
    constraint (slope < MAX_SLOPE from the onboard elevation map).
  * 3D animation: the rover drives over a risk-shaded surface with the hallucinated
    MPPI fan and chosen plan draped on the terrain, plus a top-down risk map.

Run:  python rover_3d.py      ->  writes rover3d.mp4
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
np.random.seed(1)
DEV = "cpu"

# --------------------------------------------------------------------------- #
# 1. CONFIG
# --------------------------------------------------------------------------- #
W3 = 50.0
V_MAX3, OMEGA_MAX3, DT3 = 2.2, 1.4, 0.30
START3 = np.array([5.0, 5.0, np.deg2rad(45)], np.float32)
GOAL3 = np.array([45.0, 45.0], np.float32)
GN, GS = 7, 6.0                       # egocentric sensor grid: 7x7 out to +/-6 m
OBS3 = 2 * GN * GN + 3                 # 2 channels (rel-elev, risk) + [v, pitch, roll]
LAT3, ACT3 = 40, 2
SLIP = 0.5                            # downhill gravity-slip strength
TRACT = 0.8                          # traction loss coefficient on slope
SENSOR_NOISE, PROC_NOISE = 0.02, 0.03
MAX_SLOPE = 0.85                      # hard rollover limit (|grad| ~ tan of slope angle)

# --------------------------------------------------------------------------- #
# 2. 2.5-D ELEVATION TERRAIN (analytic)
# --------------------------------------------------------------------------- #
#                cx  cy  sigma amplitude (+hill / -crater)
TERR = np.array([[25, 25, 7.0, 7.0], [15, 33, 5.0, 4.0], [35, 14, 5.0, 3.5],
                 [39, 40, 4.0, -3.0], [10, 15, 4.0, 2.5], [44, 22, 3.5, 2.5]], np.float32)
INCL = np.array([0.02, 0.02], np.float32)            # gentle global incline
ROCKS = np.array([[20, 20, 2.0, 0.9], [30, 33, 2.2, 1.0], [33, 26, 1.8, 0.8],
                  [18, 40, 2.0, 0.7], [42, 32, 2.0, 0.9], [13, 27, 1.8, 0.6]], np.float32)


def elevation(x, y):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    z = INCL[0] * x + INCL[1] * y
    for cx, cy, s, a in TERR:
        z += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    return z


def elev_grad(x, y):                                  # analytic surface gradient (slope)
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    gx = np.full(np.broadcast(x, y).shape, INCL[0], np.float32)
    gy = np.full_like(gx, INCL[1])
    for cx, cy, s, a in TERR:
        e = a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        gx += e * (-(x - cx) / (s * s)); gy += e * (-(y - cy) / (s * s))
    return gx, gy


def rough_field(x, y):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    r = np.zeros(np.broadcast(x, y).shape, np.float32)
    for cx, cy, s, p in ROCKS:
        r += p * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    return r


def slope_mag(x, y):
    gx, gy = elev_grad(x, y)
    return np.sqrt(gx * gx + gy * gy)


def traversal_cost(x, y):                             # what the rover must avoid
    return 1.2 * np.tanh(slope_mag(x, y) / 0.7) + 1.0 * rough_field(x, y)


# --------------------------------------------------------------------------- #
# 3. STOCHASTIC SLOPE-AWARE DYNAMICS (the real world) + NOMINAL MODEL
# --------------------------------------------------------------------------- #
def dyn_step(state, v, w, noise=True):
    x, y, th = state
    gx, gy = elev_grad(x, y); sl = float(np.hypot(gx, gy))
    traction = 1.0 / (1.0 + TRACT * sl)
    th2 = th + w * DT3 + (np.random.randn() * 0.02 if noise else 0.0)
    dx = v * np.cos(th) * DT3 * traction - gx * SLIP * DT3     # forward (reduced) + downhill slip
    dy = v * np.sin(th) * DT3 * traction - gy * SLIP * DT3
    if noise:
        dx += np.random.randn() * PROC_NOISE; dy += np.random.randn() * PROC_NOISE
    return np.array([np.clip(x + dx, 0, W3), np.clip(y + dy, 0, W3), th2], np.float32)


def nom_rollout(state, controls):                     # rover's nominal model (slope known, slip/noise not)
    K_, H_, _ = controls.shape
    x = np.full(K_, state[0], np.float32); y = np.full(K_, state[1], np.float32)
    th = np.full(K_, state[2], np.float32)
    xs, ys, smax = [x.copy()], [y.copy()], np.zeros(K_, np.float32)
    for t in range(H_):
        gx, gy = elev_grad(x, y); sl = np.sqrt(gx * gx + gy * gy); smax = np.maximum(smax, sl)
        tr = 1.0 / (1.0 + TRACT * sl)
        v, w = controls[:, t, 0], controls[:, t, 1]
        x = x + v * np.cos(th) * DT3 * tr; y = y + v * np.sin(th) * DT3 * tr; th = th + w * DT3
        x = np.clip(x, 0, W3); y = np.clip(y, 0, W3); xs.append(x.copy()); ys.append(y.copy())
    return np.stack(xs, 1), np.stack(ys, 1), smax


# --------------------------------------------------------------------------- #
# 4. NOISY MULTI-CHANNEL EGOCENTRIC OBSERVATION
# --------------------------------------------------------------------------- #
_o = np.linspace(-GS, GS, GN).astype(np.float32)
_gx, _gy = np.meshgrid(_o, _o); _gx, _gy = _gx.ravel(), _gy.ravel()


def observe3(state, v, noise=True):
    x, y, th = state; c, s = np.cos(th), np.sin(th)
    wx = x + c * _gx - s * _gy; wy = y + s * _gx + c * _gy
    rel_elev = (elevation(wx, wy) - elevation(x, y)) / 5.0     # terrain shape relative to rover
    risk = traversal_cost(wx, wy)
    gx, gy = elev_grad(x, y)
    pitch = gx * c + gy * s; roll = -gx * s + gy * c           # IMU-like
    obs = np.concatenate([rel_elev, risk, [v / V_MAX3, pitch, roll]]).astype(np.float32)
    if noise:
        obs = obs + np.random.randn(*obs.shape).astype(np.float32) * SENSOR_NOISE
    return obs


# --------------------------------------------------------------------------- #
# 5. JEPA NETWORKS (richer observation)
# --------------------------------------------------------------------------- #
class Enc3(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(OBS3, 192), nn.GELU(),
                               nn.Linear(192, 192), nn.GELU(), nn.Linear(192, LAT3))

    def forward(self, o):
        return self.n(o)


class Pred3(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(LAT3 + ACT3, 192), nn.GELU(),
                               nn.Linear(192, 192), nn.GELU(), nn.Linear(192, LAT3))

    def forward(self, z, a):
        return z + self.n(torch.cat([z, a], -1))


class Eval3(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(LAT3, 96), nn.GELU(),
                               nn.Linear(96, 1), nn.Softplus())

    def forward(self, z):
        return self.n(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# 6. SELF-SUPERVISED TRAINING (on noisy transitions)
# --------------------------------------------------------------------------- #
def collect3(n=700, T=14):
    O, A, O2, R = [], [], [], []
    for _ in range(n):
        st = np.array([np.random.uniform(2, 48), np.random.uniform(2, 48),
                       np.random.uniform(-np.pi, np.pi)], np.float32)
        v = np.random.uniform(0, V_MAX3)
        for _ in range(T):
            w = np.random.uniform(-OMEGA_MAX3, OMEGA_MAX3)
            v = float(np.clip(v + np.random.uniform(-0.5, 0.5), 0, V_MAX3))
            O.append(observe3(st, v))
            A.append(np.array([v / V_MAX3, w / OMEGA_MAX3], np.float32))
            st2 = dyn_step(st, v, w, noise=True)
            O2.append(observe3(st2, v)); R.append(float(traversal_cost(st[0], st[1])))
            st = st2
    t = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=DEV)
    return t(O), t(A), t(O2), t(R)


def var_reg(z):
    return F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)).mean()


def train_jepa(enc, pred, evalr, tgt, iters=1600):
    O, A, O2, R = collect3()
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()) +
                           list(evalr.parameters()), lr=1e-3)
    N = O.shape[0]
    for it in range(iters):
        idx = torch.randint(0, N, (256,), device=DEV)
        o, a, o2, r = O[idx], A[idx], O2[idx], R[idx]
        s = enc(o)
        with torch.no_grad():
            s2t = tgt(o2)
        s2p = pred(s, a)
        loss = F.mse_loss(s2p, s2t) + F.mse_loss(evalr(s), r) + var_reg(s) + var_reg(s2p)
        loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            for pt, ps in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(0.99).add_(ps, alpha=0.01)
        if it % 400 == 0 or it == iters - 1:
            print(f"  it {it:4d} loss={loss.item():.4f} latent_std={s.std(0).mean():.2f}")
    for m in (enc, pred, evalr):
        m.eval()
    with torch.no_grad():
        s = enc(O)
        corr = np.corrcoef(evalr(s).cpu().numpy(), R.cpu().numpy())[0, 1]
        pmse = F.mse_loss(pred(s, A), tgt(O2)).item(); nmse = F.mse_loss(s, tgt(O2)).item()
    print(f"[audit] risk corr(pred,true)={corr:.3f} | predictor beats no-op {nmse / pmse:.1f}x")


# --------------------------------------------------------------------------- #
# 7. MPPI: JEPA risk + hard rollover safety constraint
# --------------------------------------------------------------------------- #
K3, H3 = 120, 10
SIG3 = np.array([0.5, 0.6], np.float32)
WG, WR, WC, WSAFE, TEMP3 = 1.0, 2.6, 0.02, 40.0, 0.15


@torch.no_grad()
def mppi3(state, v_cur, U, enc, pred, evalr):
    eps = np.random.randn(K3, H3, 2).astype(np.float32) * SIG3
    V = U[None] + eps
    V[:, :, 0] = np.clip(V[:, :, 0], 0, V_MAX3); V[:, :, 1] = np.clip(V[:, :, 1], -OMEGA_MAX3, OMEGA_MAX3)
    xs, ys, smax = nom_rollout(state, V)
    goal = np.sqrt((xs[:, -1] - GOAL3[0]) ** 2 + (ys[:, -1] - GOAL3[1]) ** 2)
    safe = np.maximum(0.0, smax - MAX_SLOPE)          # hard rollover penalty (onboard map)
    s = enc(torch.tensor(observe3(state, v_cur, noise=False), device=DEV)).repeat(K3, 1)
    Vt = torch.tensor(np.stack([V[:, :, 0] / V_MAX3, V[:, :, 1] / OMEGA_MAX3], -1), device=DEV)
    risk = torch.zeros(K3, device=DEV)
    for t in range(H3):
        s = pred(s, Vt[:, t, :]); risk = risk + evalr(s)
    risk = risk.cpu().numpy()
    S = WG * goal + WR * risk + WC * (V[:, :, 1] ** 2).sum(1) + WSAFE * safe
    Sn = (S - S.min()) / (S.max() - S.min() + 1e-9)
    w = np.exp(-Sn / TEMP3); w /= w.sum()
    return (w[:, None, None] * V).sum(0), V, xs, ys, w


# --------------------------------------------------------------------------- #
# 8. CLOSED-LOOP SIM UNDER STOCHASTIC DYNAMICS
# --------------------------------------------------------------------------- #
def _record(state, v_cur, xs, ys, w, ox, oy, keep):
    return dict(state=state.copy(), v=v_cur, sx=xs[keep], sy=ys[keep], sw=w[keep] / w.max(),
                ox=ox[0], oy=oy[0], risk=float(traversal_cost(state[0], state[1])),
                slope=float(slope_mag(state[0], state[1])))


def simulate(enc, pred, evalr, max_steps=260):
    state, v_cur = START3.copy(), 0.0
    U = np.zeros((H3, 2), np.float32); U[:, 0] = 1.0
    hist, reached = [], False
    for _ in range(max_steps):
        U, V, xs, ys, w = mppi3(state, v_cur, U, enc, pred, evalr)
        ox, oy, _ = nom_rollout(state, U[None]); keep = np.argsort(w)[-60:]
        hist.append(_record(state, v_cur, xs, ys, w, ox, oy, keep))
        v_cur, w0 = float(U[0, 0]), float(U[0, 1])
        state = dyn_step(state, v_cur, w0, noise=True)
        U = np.roll(U, -1, 0); U[-1] = U[-2]
        if np.hypot(state[0] - GOAL3[0], state[1] - GOAL3[1]) < 2.0:
            hist.append(_record(state, v_cur, xs, ys, w, ox, oy, keep)); reached = True; break
    return hist, reached


# --------------------------------------------------------------------------- #
# 9. 3D ANIMATION
# --------------------------------------------------------------------------- #
def animate(hist, out="rover3d.mp4"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, cm
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    p3 = np.array([h["state"] for h in hist])
    gn = 64
    gx = np.linspace(0, W3, gn); gy = np.linspace(0, W3, gn); GX, GY = np.meshgrid(gx, gy)
    Z = elevation(GX, GY); Cc = traversal_cost(GX, GY); Cn = Cc / Cc.max()
    fcolors = cm.inferno(Cn)
    zof = lambda x, y: elevation(x, y) + 0.35

    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(GX, GY, Z, facecolors=fcolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, alpha=0.97, shade=False)
    ax.set_xlim(0, W3); ax.set_ylim(0, W3); ax.set_zlim(float(Z.min()), float(Z.max()) + 2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("elev (m)")
    ax.set_title("3D terrain — surface colour = slip/roughness risk")
    ax.plot([START3[0]], [START3[1]], [zof(*START3[:2])], "o", color="deepskyblue", ms=8, mec="white")
    ax.plot([GOAL3[0]], [GOAL3[1]], [zof(*GOAL3)], "*", color="lime", ms=16, mec="black")
    samp3 = Line3DCollection([], linewidths=0.6); ax.add_collection3d(samp3, autolim=False)
    opt3, = ax.plot([], [], [], color="cyan", lw=2.5)
    trail3, = ax.plot([], [], [], color="white", lw=2)
    rover3, = ax.plot([], [], [], "^", color="red", ms=9, mec="white")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(Cc, extent=[0, W3, 0, W3], origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02).set_label("traversal cost (slope + roughness)")
    ax2.contour(GX, GY, Cc, levels=[0.4, 0.8, 1.2], colors="white", linewidths=0.4, alpha=0.4)
    ax2.plot(*START3[:2], "o", color="deepskyblue", ms=10, mec="white", label="start")
    ax2.plot(*GOAL3, "*", color="lime", ms=18, mec="black", label="goal")
    ax2.add_patch(plt.Circle(GOAL3, 2.0, fill=False, ec="lime", ls="--"))
    samp2 = LineCollection([], linewidths=0.7); ax2.add_collection(samp2)
    opt2, = ax2.plot([], [], color="cyan", lw=2.5, label="MPPI plan")
    trail2, = ax2.plot([], [], color="white", lw=2, label="executed path")
    rov2 = plt.Polygon(np.zeros((3, 2)), closed=True, fc="red", ec="white", lw=1.2); ax2.add_patch(rov2)
    ax2.set_xlim(0, W3); ax2.set_ylim(0, W3); ax2.set_aspect("equal"); ax2.set_title("top-down risk map")
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.6)
    hud = ax2.text(1, 48.5, "", color="white", fontsize=9, va="top",
                   bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    def tri(x, y, th, L=1.8, Wd=1.1):
        pts = np.array([[L, 0], [-0.6 * L, Wd], [-0.6 * L, -Wd]], np.float32)
        c, s = np.cos(th), np.sin(th)
        return pts @ np.array([[c, s], [-s, c]], np.float32) + [x, y]

    def cols(n, sw):
        c = np.zeros((n, 4)); c[:, 0] = 0.3; c[:, 1] = 0.9; c[:, 2] = 1.0
        c[:, 3] = 0.04 + 0.45 * sw
        return c

    def update(i):
        h = hist[i]; x, y, th = h["state"]
        segs3 = [np.column_stack([h["sx"][k], h["sy"][k], zof(h["sx"][k], h["sy"][k])])
                 for k in range(len(h["sx"]))]
        samp3.set_segments(segs3); samp3.set_color(cols(len(segs3), h["sw"]))
        opt3.set_data(h["ox"], h["oy"]); opt3.set_3d_properties(zof(np.asarray(h["ox"]), np.asarray(h["oy"])))
        trail3.set_data(p3[:i + 1, 0], p3[:i + 1, 1]); trail3.set_3d_properties(zof(p3[:i + 1, 0], p3[:i + 1, 1]))
        rover3.set_data([x], [y]); rover3.set_3d_properties([zof(x, y)])
        ax.view_init(elev=46, azim=-60 + 0.18 * i)
        segs2 = [np.column_stack([h["sx"][k], h["sy"][k]]) for k in range(len(h["sx"]))]
        samp2.set_segments(segs2); samp2.set_color(cols(len(segs2), h["sw"]))
        opt2.set_data(h["ox"], h["oy"]); trail2.set_data(p3[:i + 1, 0], p3[:i + 1, 1])
        rov2.set_xy(tri(x, y, th))
        d = np.hypot(x - GOAL3[0], y - GOAL3[1])
        hud.set_text(f"step {i}\nv = {h['v']:.2f} m/s\nrisk = {h['risk']:.2f}\n"
                     f"slope = {h['slope']:.2f}\nto goal = {d:.1f} m")
        return ()

    ani = animation.FuncAnimation(fig, update, frames=len(hist), interval=85)
    ani.save(out, writer="ffmpeg", dpi=110, fps=12)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main():
    enc, pred, evalr = Enc3().to(DEV), Pred3().to(DEV), Eval3().to(DEV)
    tgt = Enc3().to(DEV)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)

    print("training 3D JEPA world-model on noisy transitions ...")
    train_jepa(enc, pred, evalr, tgt)
    print("running closed-loop MPPI under stochastic slope dynamics ...")
    hist, reached = simulate(enc, pred, evalr)
    p3 = np.array([h["state"] for h in hist])
    rk = np.array([h["risk"] for h in hist]); sp = np.array([h["slope"] for h in hist])
    print(f"frames={len(hist)} reached={reached} "
          f"final_dist={np.hypot(p3[-1, 0] - GOAL3[0], p3[-1, 1] - GOAL3[1]):.2f} m")
    print(f"risk(mean={rk.mean():.3f},max={rk.max():.3f}) slope(mean={sp.mean():.3f},"
          f"max={sp.max():.3f}/limit {MAX_SLOPE}) rollover_breaches={(sp > MAX_SLOPE).sum()}")
    out = animate(hist)
    print(f"saved {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
