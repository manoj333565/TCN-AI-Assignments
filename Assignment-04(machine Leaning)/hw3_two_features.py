"""Homework 3 (Stretch, ~15 min): Two features by hand.

Goal: extend gradient descent from one feature (y_hat = w*x + b) to TWO
features (y_hat = w1*x1 + w2*x2 + b) using a weight VECTOR and the matrix
product X @ w. Generate a second feature on a wildly different scale
(x2 ~ 0..1000, like "monthly hours") and show TWO things:

  1. WITHOUT scaling: gradient descent diverges (loss -> nan/inf) for any
     learning rate big enough to move w1 at all, because the narrow valley
     in the w2 direction forces a tiny lr that barely moves w1.
  2. WITH StandardScaler: both features share a scale, the valley becomes a
     round bowl, and a large lr converges in a few epochs. This is exactly
     why we always scale before gradient descent — and it sets up Day 05.

Run:  py hw3_two_features.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


# --- same x1 (experience) as class, plus a big-scale x2 -----------------------
rng = np.random.default_rng(4)
x1 = np.round(rng.uniform(0, 10, 30), 1)               # years of experience
x2 = rng.uniform(0, 1000, 30)                          # "monthly hours" 0..1000
X = np.column_stack([x1, x2])                           # shape (30, 2)
y = 3 + 1.8 * x1 + 0.004 * x2 + rng.normal(0, 1.2, 30)  # true weights: w1=1.8, w2=0.004


def gd_multi(X, y, lr, epochs):
    """Vectorised gradient descent for y_hat = X @ w + b.

    w is now a VECTOR (one weight per feature). X @ w gives all predictions
    in one shot. The gradient for every weight at once is X.T @ err / m.
    """
    w = np.zeros(X.shape[1])                            # [w1, w2], starts at 0
    b = 0.0
    for _ in range(epochs):
        err = X @ w + b - y                             # (30,)  predictions - actuals
        w -= lr * (X.T @ err) / len(y)                  # gradient for BOTH weights
        b -= lr * err.mean()
    final_loss = ((X @ w + b - y) ** 2).mean() / 2
    return w, b, final_loss


# --- 1. RAW features: diverges -----------------------------------------------
# x2 ~ 1000 makes the w2-gradient huge, so any lr that would move w1
# (say 1e-3) sends w2 to infinity. We suppress the inevitable overflow warning.
with np.errstate(all="ignore"):
    w_raw, b_raw, J_raw = gd_multi(X, y, lr=1e-3, epochs=500)

if np.isfinite(J_raw):
    print(f"raw  (lr=1e-3): converged to J={J_raw:.3f}  (unexpected — try more epochs)")
else:
    print("raw  (lr=1e-3): DIVERGED (nan/inf) — x2 ~ 1000 makes the valley too narrow.")
    print("                   A lr small enough to be safe for w2 (1e-6) would barely")
    print("                   move w1, so learning stalls. Scale is the fix.")

# Show that a tiny lr is safe but stalls:
w_tiny, _, J_tiny = gd_multi(X, y, lr=1e-6, epochs=500)
print(f"raw  (lr=1e-6): J={J_tiny:.3f}  (no divergence, but w1={w_tiny[0]:.4f} barely moved "
      f"— true w1=1.8)")


# --- 2. SCALED features: converges fast --------------------------------------
scaler = StandardScaler().fit(X)                       # learn mean & std on X
Xs = scaler.transform(X)                               # both columns now ~ N(0,1)
w_s, b_s, J_s = gd_multi(Xs, y, lr=0.5, epochs=100)
print(f"\nscaled (lr=0.5): J={J_s:.3f}  converged!  w_scaled={w_s.round(3)}")
print("  (weights are on the SCALED features; inverse-transform to read them on the")
print("   original scale, but the point is: scaling let a 500x larger lr converge.)")

# Read the weights back on the original scale: w_orig_j = w_scaled_j / std_j
w_orig = w_s / scaler.scale_
b_orig = b_s - (w_s * scaler.mean_ / scaler.scale_).sum()
print(f"\n  recovered original weights: w1={w_orig[0]:.3f} (true 1.8), "
      f"w2={w_orig[1]:.4f} (true 0.004), b={b_orig:.3f} (true 3.0)")
