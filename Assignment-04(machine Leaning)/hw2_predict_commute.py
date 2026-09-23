"""Homework 2 (Medium, ~15 min, daily life): Predict your commute.

Goal: take 15 real trips (distance in km vs minutes taken), fit a line TWO ways
— with your own gradient descent AND with sklearn's LinearRegression — and
compare the slope (w) and intercept (b). The slope is your "minutes per km";
the intercept is the fixed overhead (waiting / walking / boarding time).

A sample of 15 trips is generated below with a fixed seed so the script runs
out of the box. Replace `distance` and `minutes` with your own 15 numbers to
make it personal.

Run:  py hw2_predict_commute.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# --- 15 trips: distance (km) vs minutes taken --------------------------------
# Replace these with your real data.  Sample is seeded so it reproduces.
rng = np.random.default_rng(7)
distance = np.round(rng.uniform(2, 25, 15), 1)          # km
# True rule we pretend: minutes = 8 + 3.2*km + noise (3.2 min per km)
minutes = np.round(8 + 3.2 * distance + rng.normal(0, 3, 15), 1)


# --- 1. our own gradient descent (same function as class) -------------------
# Scale x first (the Day 04 lesson: scaling makes GD converge fast and stable).
# We then unscale the learned weights to read them in real units (min per km).
def predict(x, w, b):
    return w * x + b

def gradient_descent(x, y, lr, epochs, w0=0.0, b0=0.0):
    w, b = w0, b0
    for _ in range(epochs):
        err = predict(x, w, b) - y
        w -= lr * (err * x).mean()
        b -= lr * err.mean()
    return w, b

x_mean, x_std = distance.mean(), distance.std()
x_scaled = (distance - x_mean) / x_std                  # ~N(0,1), GD loves this
w_s, b_s = gradient_descent(x_scaled, minutes, lr=0.1, epochs=3000)

# Unscale: the line y = w_s * x_scaled + b_s is the same as
# y = (w_s / x_std) * x + (b_s - w_s * x_mean / x_std)
w_gd = w_s / x_std
b_gd = b_s - w_s * x_mean / x_std


# --- 2. sklearn's closed-form solution ---------------------------------------
X = distance.reshape(-1, 1)                              # sklearn wants 2-D
sk = LinearRegression().fit(X, minutes)
w_sk, b_sk = sk.coef_[0], sk.intercept_


# --- compare ------------------------------------------------------------------
print("            w (min/km)   b (overhead min)")
print(f"  our GD  :   {w_gd:7.3f}      {b_gd:7.3f}")
print(f"  sklearn :   {w_sk:7.3f}      {b_sk:7.3f}")
print(f"\n  -> Every extra km adds about {w_sk:.1f} minutes.")
print(f"  -> b = {b_sk:.1f} min is the fixed cost: waiting, walking, boarding.")

# --- plot both lines on the data ---------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(distance, minutes, color="tab:blue", label="your trips")
xs = np.linspace(0, distance.max() + 1, 2)
ax.plot(xs, predict(xs, w_gd, b_gd), "--", color="tab:orange", label="our GD")
ax.plot(xs, predict(xs, w_sk, b_sk), color="tab:green", label="sklearn")
ax.set_xlabel("distance (km)")
ax.set_ylabel("minutes")
ax.set_title("Commute: minutes vs distance")
ax.legend()
plt.tight_layout()
fig.savefig("commute_fit.png", dpi=150)
print("\nsaved commute_fit.png")
plt.show()
