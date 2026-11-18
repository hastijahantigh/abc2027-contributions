"""
Amir's parameter-free physics features, per radar and timestep, on the z-scored CIR of a whole
stream (so a window's features never depend on where the window starts):

    motion       mean over range bins of the squared difference between consecutive *raw* frames
                 (a run of identical 50 Hz frames is one raw frame, timed at the run's centre, as in
                 the official labeller), interpolated onto the grid and averaged over 0.2 s
    step_m       mean over bins of (mean profile after - mean profile before)^2, the two sides
                 m x 0.2 s long and 0.2 s apart (the 0.2 s token between them, as in Amir's tokens),
                 for m = 1, 5, 15 (0.2, 1 and 3 s)

Each value f becomes 0.3 log(f + 1e-3) + 1 (Amir's scaling). A side that leaves the stream is
truncated; with no timestep on one side the step is 0 before scaling. Output: [12, T] float32,
radar-major (motion, step_1, step_5, step_15 of radar 0, then radar 1, then radar 2).
"""
import numpy as np

TOKEN = 10                      # 0.2 s at 50 Hz
STEP_SCALES = (1, 5, 15)
LOG_EPS = 1e-3
NUM_FEATURES = 1 + len(STEP_SCALES)


def scale(f: np.ndarray) -> np.ndarray:
    return 0.3 * np.log(f + LOG_EPS) + 1.0


def raw_motion(z: np.ndarray) -> np.ndarray:
    """z [T, bins] of one radar -> mean squared raw-frame difference per bin on the grid [T]."""
    T = len(z)
    change = np.ones(T, bool)
    change[1:] = np.any(z[1:] != z[:-1], axis=1)
    starts = np.flatnonzero(change)
    if len(starts) < 3:
        return np.zeros(T)
    stops = np.r_[starts[1:], T]
    times = (starts + stops - 1) / 2.0
    delta = np.diff(z[starts].astype(np.float64), axis=0)
    motion = np.einsum("ij,ij->i", delta, delta) / z.shape[1]
    return np.interp(np.arange(T, dtype=np.float64), times[1:], motion, left=float(motion[0]), right=float(motion[-1]))


def _moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    T = len(x)
    lo = np.clip(np.arange(T) - width // 2, 0, T)
    hi = np.clip(np.arange(T) - width // 2 + width, 0, T)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def profile_steps(z: np.ndarray, scales=STEP_SCALES) -> np.ndarray:
    """z [T, bins] of one radar -> [len(scales), T] mean squared step of the mean profile."""
    T, bins = z.shape
    c = np.zeros((T + 1, bins))
    np.cumsum(z.astype(np.float64), axis=0, out=c[1:])
    t = np.arange(T)
    half = TOKEN // 2
    out = np.zeros((len(scales), T))
    for i, m in enumerate(scales):
        b1 = np.clip(t - half, 0, T)
        b0 = np.clip(b1 - m * TOKEN, 0, T)
        a0 = np.clip(t + half, 0, T)
        a1 = np.clip(a0 + m * TOKEN, 0, T)
        nb, na = b1 - b0, a1 - a0
        ok = (nb > 0) & (na > 0)
        before = (c[b1] - c[b0]) / np.maximum(nb, 1)[:, None]
        after = (c[a1] - c[a0]) / np.maximum(na, 1)[:, None]
        out[i] = np.where(ok, np.square(after - before).mean(1), 0.0)
    return out


def stream_physics(z: np.ndarray) -> np.ndarray:
    """z [3, T, bins] (z-scored CIR of a whole stream) -> scaled features [12, T] float32."""
    feats = []
    for r in range(z.shape[0]):
        feats.append(_moving_mean(raw_motion(z[r]), TOKEN))
        feats.extend(profile_steps(z[r]))
    return scale(np.stack(feats)).astype(np.float32)
