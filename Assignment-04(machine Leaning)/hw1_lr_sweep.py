"""Homework 1 (Easy, ~10 min): Learning-rate sweep.

Goal: run the SAME gradient descent for five different learning rates
(alpha = 0.001, 0.003, 0.01, 0.03, 0.05) on the raw salary data, plot all five
loss curves on ONE labelled chart with a log-scale y-axis, and write one
sentence per curve describing what it does. Save the chart as lr_sweep.png.

This reuses the exact predict / cost / gradient_descent from class.
"""

import numpy as np
import matplotlib.pyplot as plt


# --- the same data from class: 30 (experience, salary) points ----------------
rng = np.random.default_rng(4)
x = np.round(rng.uniform(0, 10, 30), 1)                # years of experience
y = np.round(3 + 1.8 * x + rng.normal(0, 1.2, 30), 2)  # salary in LPA


# --- model + cost + optimiser (copied from the Day 04 notebook) --------------
def predict(x, w, b):
    return w * x + b

def cost(x, y, w, b):
    err = predict(x, w, b) - y
    return (err ** 2).mean() / 2

def gradient_descent(x, y, lr=0.01, epochs=200, w0=0.0, b0=0.0):
    """Batch gradient descent for y_hat = w*x + b."""
    w, b = w0, b0
    J_hist = []
    for _ in range(epochs):
        err = predict(x, w, b) - y
        dw = (err * x).mean()         # dJ/dw
        db = err.mean()               # dJ/db
        w -= lr * dw
        b -= lr * db
        J_hist.append(cost(x, y, w, b))
    return w, b, J_hist


# --- sweep five learning rates on the RAW x ----------------------------------
alphas = [0.001, 0.003, 0.01, 0.03, 0.05]

fig, ax = plt.subplots(figsize=(9, 4.5))
for lr in alphas:
    _, _, J = gradient_descent(x, y, lr=lr, epochs=100)
    # clip so a diverging run (huge loss) does not squash every other curve.
    J = np.clip(J, 1e-3, 1e6)
    ax.plot(J, label=f"alpha = {lr}")

ax.set_yscale("log")                                    # log-scale y as asked
ax.set_xlabel("epoch")
ax.set_ylabel("loss J  (log scale)")
ax.set_title("Learning-rate sweep: which alpha descends fastest?")
ax.legend()
plt.tight_layout()
fig.savefig("lr_sweep.png", dpi=150)
print("saved lr_sweep.png")
plt.show()

# --- one sentence per curve (printed for the student to read) -----------------
notes = {
    0.001: "Crawls down very slowly — too cautious, barely moved in 100 epochs.",
    0.003: "Faster than 0.001 but still sluggish; safe but wasteful.",
    0.01:  "Steady, smooth descent — a good default for this data.",
    0.03:  "Falls quickest and stays low — the best alpha here.",
    0.05:  "Oscillates / bounces around the bottom; almost too large.",
}
print("\nOne sentence per curve:")
for lr in alphas:
    print(f"  alpha={lr}: {notes[lr]}")
