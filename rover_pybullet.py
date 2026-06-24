"""
JEPA + MPPI rover navigation on REAL rigid-body physics (PyBullet) — single-file.

Parts 1 & 2 (`rover_2d.py`, `rover_3d.py`) plan over *analytic* terrain and
*analytic* rover kinematics. This Part 3 drops the rover into a genuine physics
simulator and keeps the same JEPA-in-latent-space + MPPI idea:

  * A **Husky** drives over a **GEOM_HEIGHTFIELD** terrain (Gaussian hills) with
    full PyBullet contact dynamics — wheel slip, suspension, chassis pitch/roll.
  * The rover senses the world the hard way: a 7x7 **egocentric elevation patch**
    built from downward **ray casts**, plus its IMU-like pitch/roll and speed.
  * Skid-steer is weak and speed-coupled, so the planner's nominal model is
    **calibrated from the real actuator** (probe the Husky, fit forward & yaw
    gains). Residual model error is corrected by receding-horizon replanning.
  * The **JEPA** world-model is trained self-supervised on transitions collected
    from real physics rollouts; its risk head is grounded on the **real chassis
    tilt** the rover actually experienced (not an analytic label).
  * **MPPI** plans entirely in the learned latent space (imagined risk) fused with
    a hard onboard-DEM slope limit, a stay-on-map boundary, and a goal term.
  * Output: a chase-cam GIF (real physics) beside a top-down slope-risk map with
    the live MPPI fan the JEPA model hallucinates.

Run (needs `pip install pybullet`):  python rover_pybullet.py  -> rover_pybullet.gif
Built / rendered on Google Colab (CPU); see the notebook for an Open-in-Colab badge.
"""
import os, math
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"

# --------------------------------------------------------------------------- #
# 1. CONFIG
# --------------------------------------------------------------------------- #
WORLD   = 24.0                 # terrain is WORLD x WORLD metres, centred on origin
NHF     = 128                  # heightfield resolution
MESH    = WORLD / NHF
START   = np.array([-9.0, -9.0, np.deg2rad(45)], np.float32)   # x, y, heading
GOAL    = np.array([ 9.0,  9.0], np.float32)
V_MAX, OMEGA_MAX = 3.0, 3.0
GN, GS  = 7, 2.5               # 7x7 ego elevation patch out to +/- 2.5 m
OBS     = GN * GN + 3          # rel-elevation patch + [v, pitch, roll]
LAT, ACT = 40, 2
WHEEL_R, TRACK = 0.165, 0.555  # husky-ish wheel radius / track
STEER   = 1.3                  # COMMANDED skid-steer differential (> track => aggressive yaw)
WHEEL_F = 120                  # per-wheel motor force
STEP    = 12                   # sim substeps per control tick
MAX_TILT = 0.45                # rad — onboard-DEM slope limit the planner respects
ROLLOVER = 0.80                # rad — genuine flip threshold (for reporting only)

# Gaussian hills (cx, cy, sigma, amplitude) in world metres ; +hill / -dip
HILLS = np.array([[ 0.0,  0.0, 3.0,  2.6],
                  [-5.0,  4.0, 2.2,  1.7],
                  [ 5.5, -3.0, 2.4,  1.5],
                  [ 4.0,  6.0, 2.0, -1.2],
                  [-6.0, -5.0, 2.2,  1.3]], np.float32)


def elevation(x, y):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    z = np.zeros(np.broadcast(x, y).shape, np.float32)
    for cx, cy, s, a in HILLS:
        z += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    return z


def elev_grad(x, y):                                   # analytic surface gradient = onboard DEM
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    gx = np.zeros(np.broadcast(x, y).shape, np.float32); gy = gx.copy()
    for cx, cy, s, a in HILLS:
        e = a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        gx = gx + e * (-(x - cx) / (s * s)); gy = gy + e * (-(y - cy) / (s * s))
    return gx, gy


def slope_mag(x, y):
    gx, gy = elev_grad(x, y)
    return np.sqrt(gx * gx + gy * gy)                  # |grad| ~ tan(tilt)


# --------------------------------------------------------------------------- #
# 2. THE REAL ENVIRONMENT (PyBullet)
# --------------------------------------------------------------------------- #
class RoverEnv:
    """Heightfield terrain + Husky with full contact dynamics, differential
    drive, and raycast egocentric elevation sensing. Every PyBullet call is
    pinned to this body's own physics client so several envs can coexist."""

    def __init__(self, gui=False):
        self.c = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.c)
        p.setGravity(0, 0, -9.81, physicsClientId=self.c)
        xs = (np.arange(NHF) - NHF / 2) * MESH
        GX, GY = np.meshgrid(xs, xs)
        self.hf = elevation(GX, GY).astype(np.float32)
        self.z0 = float((self.hf.max() + self.hf.min()) / 2)   # PyBullet centres HF on this
        shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, meshScale=[MESH, MESH, 1.0],
                                       heightfieldData=self.hf.flatten(order="C"),
                                       numHeightfieldRows=NHF, numHeightfieldColumns=NHF,
                                       physicsClientId=self.c)
        self.terrain = p.createMultiBody(0, shape, physicsClientId=self.c)
        p.resetBasePositionAndOrientation(self.terrain, [0, 0, self.z0], [0, 0, 0, 1], physicsClientId=self.c)
        p.changeVisualShape(self.terrain, -1, rgbaColor=[0.55, 0.45, 0.35, 1], physicsClientId=self.c)
        self.husky = p.loadURDF("husky/husky.urdf", [0, 0, 2], physicsClientId=self.c)
        wj = lambda j: p.getJointInfo(self.husky, j, physicsClientId=self.c)[1]
        wheels = [j for j in range(p.getNumJoints(self.husky, physicsClientId=self.c)) if b"wheel" in wj(j)]
        self.lw = [j for j in wheels if b"left"  in wj(j)]
        self.rw = [j for j in wheels if b"right" in wj(j)]

    def ground_z(self, x, y):
        hit = p.rayTest([x, y, 30], [x, y, -30], physicsClientId=self.c)[0]
        return hit[3][2] if hit[0] >= 0 else self.z0

    def reset(self, pose):
        x, y, th = pose
        z = self.ground_z(x, y) + 0.25
        q = p.getQuaternionFromEuler([0, 0, float(th)])
        p.resetBasePositionAndOrientation(self.husky, [float(x), float(y), z], q, physicsClientId=self.c)
        p.resetBaseVelocity(self.husky, [0, 0, 0], [0, 0, 0], physicsClientId=self.c)
        for _ in range(40):                              # let it settle onto the surface
            p.stepSimulation(physicsClientId=self.c)

    def drive(self, v, w, steps=STEP):                   # differential (skid-steer) velocity control
        vl = (v - w * STEER / 2) / WHEEL_R; vr = (v + w * STEER / 2) / WHEEL_R
        for j in self.lw: p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL, targetVelocity=vl, force=WHEEL_F, physicsClientId=self.c)
        for j in self.rw: p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL, targetVelocity=vr, force=WHEEL_F, physicsClientId=self.c)
        for _ in range(steps): p.stepSimulation(physicsClientId=self.c)

    def pose(self):
        (x, y, z), q = p.getBasePositionAndOrientation(self.husky, physicsClientId=self.c)
        roll, pitch, yaw = p.getEulerFromQuaternion(q)
        return np.array([x, y, z], np.float32), float(roll), float(pitch), float(yaw)

    def pose_xyyaw(self):
        (x, y, _), _, _, yaw = self.pose()
        return np.array([x, y, yaw], np.float32)

    def speed(self):
        (vx, vy, _), _ = p.getBaseVelocity(self.husky, physicsClientId=self.c)
        return float(np.hypot(vx, vy))

    def observe(self, noise=False):
        (x, y, z), roll, pitch, yaw = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        off = np.linspace(-GS, GS, GN).astype(np.float32)
        ox, oy = np.meshgrid(off, off); ox, oy = ox.ravel(), oy.ravel()
        wx = x + c * ox - s * oy; wy = y + s * ox + c * oy   # ego grid -> world
        frm = [[float(a), float(b), z + 15] for a, b in zip(wx, wy)]
        to  = [[float(a), float(b), z - 15] for a, b in zip(wx, wy)]
        hits = p.rayTestBatch(frm, to, physicsClientId=self.c)
        zc = self.ground_z(x, y)
        patch = np.array([(h[3][2] - zc) if h[0] >= 0 else 0.0 for h in hits], np.float32) / 2.0
        obs = np.concatenate([patch, [self.speed() / V_MAX, pitch, roll]]).astype(np.float32)
        if noise:
            obs = obs + np.random.randn(*obs.shape).astype(np.float32) * 0.02
        return obs

    def frame(self, w=480, h=320):                       # 3rd-person chase camera
        (x, y, z), roll, pitch, yaw = self.pose()
        cam = [x - 4.2 * math.cos(yaw), y - 4.2 * math.sin(yaw), z + 2.6]
        view = p.computeViewMatrix(cam, [x, y, z], [0, 0, 1], physicsClientId=self.c)
        proj = p.computeProjectionMatrixFOV(60, w / h, 0.1, 60, physicsClientId=self.c)
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.c)[2]
        return np.reshape(img, (h, w, 4))[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect(physicsClientId=self.c)


# --------------------------------------------------------------------------- #
# 3. CALIBRATED NOMINAL MODEL (what the planner believes)
# --------------------------------------------------------------------------- #
# Skid-steer yaw authority is weak and dies with forward speed, so we MEASURE the
# real actuator and fit a unicycle: forward = KF*v, yaw = (KW0 - KWV*v)*w  (per tick).
KF, KW0, KWV = 0.05, 0.04, 0.01      # placeholders; overwritten by calibrate()
FLAT = np.array([-11.0, 2.0, 0.0], np.float32)   # near-flat corner for calibration


def calibrate(env, warm=8, hold=16):
    global KF, KW0, KWV

    def meas(v, w):
        env.reset(FLAT)
        for _ in range(warm): env.drive(v, w)
        (x0, y0, _), _, _, a0 = env.pose()
        for _ in range(hold): env.drive(v, w)
        (x1, y1, _), _, _, a1 = env.pose()
        fwd = ((x1 - x0) * math.cos(a0) + (y1 - y0) * math.sin(a0)) / hold        # m / tick
        dyaw = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)) / hold            # rad / tick
        return fwd, dyaw

    KF = float(np.mean([meas(v, 0.0)[0] / v for v in (1.0, 2.0, 3.0)]))
    samples = [(v, meas(v, w)[1] / w) for v in (0.0, 1.0, 2.0) for w in (1.5, 3.0)]
    V = np.array([s[0] for s in samples]); G = np.array([s[1] for s in samples])
    a = np.polyfit(V, G, 1); KWV, KW0 = -float(a[0]), float(a[1])
    print(f"[calibrate] KF={KF:.4f} m/tick/v   yaw gain(v)={KW0:.4f} - {KWV:.4f}*v  rad/tick/w")


def nom_rollout(state, controls):
    """state=(x,y,yaw); controls (K,H,2) actual (v,w). Returns per-sample x,y paths.
    Purely kinematic and terrain-blind — the JEPA model supplies the terrain risk."""
    K_, H_, _ = controls.shape
    x = np.full(K_, state[0], np.float32); y = np.full(K_, state[1], np.float32)
    th = np.full(K_, state[2], np.float32)
    xs, ys = [x.copy()], [y.copy()]
    for t in range(H_):
        v = controls[:, t, 0]; w = controls[:, t, 1]
        gain = np.clip(KW0 - KWV * v, 0.0, None)         # yaw authority dies at speed
        x = np.clip(x + KF * v * np.cos(th), -12, 12)
        y = np.clip(y + KF * v * np.sin(th), -12, 12)
        th = th + gain * w
        xs.append(x.copy()); ys.append(y.copy())
    return np.stack(xs, 1), np.stack(ys, 1)


# --------------------------------------------------------------------------- #
# 4. JEPA WORLD-MODEL
# --------------------------------------------------------------------------- #
class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(OBS, 192), nn.GELU(),
                            nn.Linear(192, 192), nn.GELU(), nn.Linear(192, LAT))
    def forward(s, o): return s.n(o)


class Pred(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(LAT + ACT, 192), nn.GELU(),
                            nn.Linear(192, 192), nn.GELU(), nn.Linear(192, LAT))
    def forward(s, z, a): return z + s.n(torch.cat([z, a], -1))   # residual latent transition


class Eval(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(LAT, 96), nn.GELU(),
                            nn.Linear(96, 1), nn.Softplus())
    def forward(s, z): return s.n(z).squeeze(-1)


def tilt(roll, pitch): return float(math.hypot(roll, pitch))      # grounded risk = real chassis tilt
def var_reg(z): return F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)).mean()   # VICReg anti-collapse


def collect(env, rollouts=90, T=12):
    """Drive the real Husky with random controls from random poses, recording
    (obs, action, next-obs, experienced-tilt) transitions."""
    O, A, O2, R = [], [], [], []
    for _ in range(rollouts):
        st = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10),
                       np.random.uniform(-np.pi, np.pi)], np.float32)
        env.reset(st); v = np.random.uniform(0, V_MAX)
        for _ in range(T):
            w = np.random.uniform(-OMEGA_MAX, OMEGA_MAX)
            v = float(np.clip(v + np.random.uniform(-0.6, 0.6), 0, V_MAX))
            O.append(env.observe(noise=True))
            A.append(np.array([v / V_MAX, w / OMEGA_MAX], np.float32))
            env.drive(v, w)
            _, roll, pitch, _ = env.pose()
            O2.append(env.observe(noise=True)); R.append(tilt(roll, pitch))
    t = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=DEV)
    return t(O), t(A), t(O2), t(R)


def train_jepa(O, A, O2, R, iters=1600):
    enc, pred, evalr = Enc().to(DEV), Pred().to(DEV), Eval().to(DEV)
    tgt = Enc().to(DEV); tgt.load_state_dict(enc.state_dict())
    for q in tgt.parameters(): q.requires_grad_(False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()) +
                           list(evalr.parameters()), lr=1e-3)
    N = O.shape[0]
    for it in range(iters):
        idx = torch.randint(0, N, (256,), device=DEV)
        o, a, o2, r = O[idx], A[idx], O2[idx], R[idx]
        s = enc(o)
        with torch.no_grad(): s2t = tgt(o2)
        s2p = pred(s, a)
        loss = F.mse_loss(s2p, s2t) + F.mse_loss(evalr(s), r) + var_reg(s) + var_reg(s2p)
        loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            for pt, ps in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(0.99).add_(ps, alpha=0.01)       # EMA target encoder
        if it % 400 == 0 or it == iters - 1:
            print(f"  it {it:4d} loss={loss.item():.4f} latent_std={s.std(0).mean():.2f}")
    for m in (enc, pred, evalr): m.eval()
    with torch.no_grad():
        s = enc(O)
        corr = np.corrcoef(evalr(s).cpu().numpy(), R.cpu().numpy())[0, 1]
        pmse = F.mse_loss(pred(s, A), tgt(O2)).item(); nmse = F.mse_loss(s, tgt(O2)).item()
    print(f"[audit] risk corr(pred,true)={corr:.3f} | predictor beats no-op {nmse / pmse:.1f}x")
    return enc, pred, evalr


# --------------------------------------------------------------------------- #
# 5. MPPI — latent risk + hard slope limit + stay-on-map, over the nominal model
# --------------------------------------------------------------------------- #
K, H = 160, 22
SIG = np.array([0.7, 1.4], np.float32)                # exploration std in (v, w)
WG, WR, WC, WSAFE, WBND, TEMP = 1.25, 10.0, 0.05, 300.0, 120.0, 0.08
TILT_LIM = math.tan(MAX_TILT)                         # |grad| rollover limit
EDGE = 10.5                                           # keep this far inside the terrain


@torch.no_grad()
def mppi(env, state, U, enc, pred, evalr):
    eps = np.random.randn(K, H, 2).astype(np.float32) * SIG
    Vc = U[None] + eps
    Vc[:, :, 0] = np.clip(Vc[:, :, 0], 0, V_MAX); Vc[:, :, 1] = np.clip(Vc[:, :, 1], -OMEGA_MAX, OMEGA_MAX)
    xs, ys = nom_rollout(state, Vc)                                  # (K, H+1)
    goal = np.hypot(xs[:, 1:] - GOAL[0], ys[:, 1:] - GOAL[1]).mean(1)            # running goal progress
    safe = np.maximum(0.0, slope_mag(xs[:, 1:], ys[:, 1:]) - TILT_LIM).mean(1)   # avoid steep ground
    bnd  = np.maximum(0.0, np.maximum(np.abs(xs[:, 1:]), np.abs(ys[:, 1:])) - EDGE).mean(1)  # stay on map
    s = enc(torch.tensor(env.observe(noise=False), device=DEV)).repeat(K, 1)
    An = torch.tensor(np.stack([Vc[:, :, 0] / V_MAX, Vc[:, :, 1] / OMEGA_MAX], -1), device=DEV)
    risk = torch.zeros(K, device=DEV)
    for t in range(H):                                               # hallucinate H steps in latent space
        s = pred(s, An[:, t, :]); risk = risk + evalr(s)
    risk = (risk / H).cpu().numpy()
    S = WG * goal + WR * risk + WC * (Vc[:, :, 1] ** 2).mean(1) + WSAFE * safe + WBND * bnd
    Sn = (S - S.min()) / (S.max() - S.min() + 1e-9)
    wts = np.exp(-Sn / TEMP); wts /= wts.sum()
    return (wts[:, None, None] * Vc).sum(0), xs, ys, wts


def run(env, enc, pred, evalr, max_ticks=440, capture=True, cap_every=3):
    env.reset(START)
    U = np.zeros((H, 2), np.float32); U[:, 0] = 1.8
    frames, cap_idx, hist, reached = [], [], [], False
    for k in range(max_ticks):
        U, xs, ys, wts = mppi(env, env.pose_xyyaw(), U, enc, pred, evalr)
        v0, w0 = float(U[0, 0]), float(U[0, 1])
        env.drive(v0, w0)
        (x, y, z), roll, pitch, yaw = env.pose()
        hist.append(dict(x=float(x), y=float(y), yaw=float(yaw), v=v0, w=w0, tilt=tilt(roll, pitch),
                         slope=float(slope_mag(x, y)), sx=xs, sy=ys, wts=wts,
                         d=float(np.hypot(x - GOAL[0], y - GOAL[1]))))
        if capture and k % cap_every == 0: frames.append(env.frame()); cap_idx.append(k)
        U = np.roll(U, -1, 0); U[-1] = U[-2]
        if hist[-1]['d'] < 1.2: reached = True; break
    return frames, cap_idx, hist, reached


# --------------------------------------------------------------------------- #
# 6. VISUALISATION — chase-cam (real physics) + top-down MPPI fan
# --------------------------------------------------------------------------- #
def make_gif(frames, cap_idx, hist, out="rover_pybullet.gif", fps=14, dpi=78):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import LineCollection

    gx = np.linspace(-12, 12, 130); GX, GY = np.meshgrid(gx, gx)
    RISK = slope_mag(GX, GY)
    px = np.array([[h['x'], h['y']] for h in hist])

    fig = plt.figure(figsize=(11.4, 4.6))
    axc = fig.add_subplot(1, 2, 1); axc.axis("off")
    axc.set_title("PyBullet chase-cam — real rigid-body physics", fontsize=10)
    im = axc.imshow(frames[0])

    ax = fig.add_subplot(1, 2, 2)
    hm = ax.imshow(RISK, extent=[-12, 12, -12, 12], origin="lower", cmap="inferno", vmax=0.9)
    fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.02).set_label("onboard-DEM slope |grad z| (rollover risk)", fontsize=8)
    ax.contour(GX, GY, elevation(GX, GY), levels=8, colors="white", linewidths=0.3, alpha=0.35)
    ax.plot(*START[:2], "o", color="deepskyblue", ms=9, mec="white", label="start")
    ax.plot(*GOAL, "*", color="lime", ms=16, mec="black", label="goal")
    ax.add_patch(plt.Circle(GOAL, 1.2, fill=False, ec="lime", ls="--", lw=1))
    fan = LineCollection([], linewidths=0.5); ax.add_collection(fan)
    trail, = ax.plot([], [], color="white", lw=2.0, label="executed path")
    rov = plt.Polygon(np.zeros((3, 2)), closed=True, fc="red", ec="white", lw=1.2); ax.add_patch(rov)
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal")
    ax.set_title("top-down — MPPI fan hallucinated by JEPA", fontsize=10)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
    hud = ax.text(-11.3, 11, "", color="white", fontsize=8, va="top",
                  bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    def tri(x, y, th, L=1.1, Wd=0.7):
        pts = np.array([[L, 0], [-0.6 * L, Wd], [-0.6 * L, -Wd]], np.float32)
        c, s = math.cos(th), math.sin(th)
        return pts @ np.array([[c, s], [-s, c]], np.float32) + [x, y]

    def update(i):
        k = cap_idx[i]; h = hist[k]; x, y, th = h['x'], h['y'], h['yaw']
        im.set_data(frames[i])
        keep = np.argsort(h['wts'])[-60:]                       # best 60 sampled rollouts
        segs = [np.column_stack([h['sx'][j], h['sy'][j]]) for j in keep]
        sw = h['wts'][keep] / (h['wts'][keep].max() + 1e-9)
        cols = np.zeros((len(keep), 4)); cols[:, 0] = 0.3; cols[:, 1] = 0.9; cols[:, 2] = 1.0
        cols[:, 3] = 0.05 + 0.5 * sw
        fan.set_segments(segs); fan.set_color(cols)
        trail.set_data(px[:k + 1, 0], px[:k + 1, 1]); rov.set_xy(tri(x, y, th))
        hud.set_text(f"tick {k}\nv={h['v']:.2f} m/s\ntilt={h['tilt']:.2f} rad\nto goal={h['d']:.1f} m")
        return ()

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main():
    env = RoverEnv()
    print("calibrating the nominal skid-steer model from the real actuator ...")
    calibrate(env)
    print("collecting transitions from REAL physics rollouts ...")
    O, A, O2, R = collect(env)
    print(f"  {O.shape[0]} transitions | risk(tilt) mean={R.mean():.3f} max={R.max():.3f}")
    print("training the JEPA world-model ...")
    enc, pred, evalr = train_jepa(O, A, O2, R)
    print("running closed-loop MPPI in the PyBullet world ...")
    frames, cap_idx, hist, reached = run(env, enc, pred, evalr)
    tl = np.array([h['tilt'] for h in hist]); sl = np.array([h['slope'] for h in hist])
    print(f"ticks={len(hist)} reached={reached} final_dist={hist[-1]['d']:.2f} m | frames={len(frames)}")
    print(f"REAL tilt mean={tl.mean():.3f} max={tl.max():.3f} | comfort>{MAX_TILT}: {(tl > MAX_TILT).sum()}"
          f"  rollover>{ROLLOVER}: {(tl > ROLLOVER).sum()} | onboard slope max={sl.max():.3f}/limit {TILT_LIM:.3f}")
    out = make_gif(frames, cap_idx, hist)
    print(f"saved {os.path.abspath(out)} ({os.path.getsize(out) / 1e6:.2f} MB)")
    env.close()


if __name__ == "__main__":
    main()
