# Day 04 Homework — How Machines Learn (Gradient Descent)

Three short tasks. Each is a runnable Python file (`hw1_...`, `hw2_...`,
`hw3_...`). Run them with `py hw1_lr_sweep.py`, etc. Each saves a labelled PNG.

---

## Q1 (Easy, 10 min) — Learning-rate sweep

**What the question is asking**
Run the same gradient descent five times, each with a different learning rate
(alpha = 0.001, 0.003, 0.01, 0.03, 0.05), on the raw salary data from class.
Plot all five loss curves on one chart with a log-scale y-axis, and write one
sentence describing what each curve does. Save it as `lr_sweep.png`.

**How to approach it**
Keep the `predict`, `cost`, and `gradient_descent` functions exactly as written
in class. Loop over the five alpha values; for each, run 100 epochs and save
the loss history. Plot all five histories on one axes with `ax.plot`, set the
y-axis to log scale (`ax.set_yscale("log")`), label the axes and add a legend.
Then look at each curve: does it drop fast, crawl, or bounce? Write one
sentence per curve. Clipping very large loss values (with `np.clip`) keeps a
diverging run from squashing all the other curves flat.

**How the solution does it (in words)**
The script regenerates the 30 (experience, salary) points with the same seed as
class, runs `gradient_descent` for each of the five alphas over 100 epochs, and
plots all five loss curves on one log-scale chart. The output shows that
alpha=0.001 crawls (too cautious), 0.003 is better but still slow, 0.01 is a
steady smooth descent, 0.03 drops fastest and stays low (the best here), and
0.05 oscillates near the bottom (almost too large). The chart is saved as
`lr_sweep.png`.

---

## Q2 (Medium, 15 min, daily life) — Predict your commute

**What the question is asking**
Record (or estimate) 15 real trips you take — the distance in km and the
minutes it took (bus, bike, metro, whatever you use). Fit a straight line
through those points two ways: once with your own gradient descent and once
with sklearn's `LinearRegression`. Compare the slope (w) and intercept (b).
The slope is your "minutes per km"; what does the intercept (b) mean
physically? (Hint: it is the fixed overhead — waiting, walking, boarding.)

**How to approach it**
Put your 15 distances in one array and the 15 travel times in another. For
your own gradient descent, scale the distance first (subtract the mean, divide
by the standard deviation) so the learning rate works well — this is the Day 04
lesson. Run gradient descent with a moderate lr (like 0.1) for a few thousand
epochs. To read the weights back in real units (minutes per km), unscale: divide
the learned weight by the standard deviation of distance, and adjust the
intercept accordingly. For sklearn, reshape the distance to a 2-D array and call
`LinearRegression().fit`. Both should give nearly identical w and b. The slope
tells you how many minutes each extra kilometre costs; the intercept is the
fixed part of every trip (the time you spend before you even start moving).

**How the solution does it (in words)**
The script generates 15 sample trips (seeded so it reproduces) with a true rule
of ~3.2 minutes per km plus an 8-minute overhead. It scales the distance,
runs gradient descent with lr=0.1 for 3000 epochs, then unscales the weights
back to real units. It also fits sklearn's `LinearRegression` on the raw
distance. Both give w = 3.21 min/km and b = 5.76 min — a perfect match. The
script prints both, explains that b is the fixed overhead (waiting, walking,
boarding), saves a scatter plot with both fitted lines as `commute_fit.png`.
Replace the sample arrays with your own 15 numbers to make it personal.

---

## Q3 (Stretch, 15 min) — Two features by hand

**What the question is asking**
Extend gradient descent from one feature (y_hat = w*x + b) to two features
(y_hat = w1*x1 + w2*x2 + b). Use a weight VECTOR and the matrix product `X @ w`
instead of a single `w * x`. Generate a second feature on a very different
scale (x2 between 0 and 1000, like "monthly hours"). Show two things: (1)
without scaling, gradient descent diverges (loss goes to infinity) for any
learning rate big enough to move w1, and (2) with `StandardScaler`, the same
gradient descent converges easily. This is the setup for Day 05.

**How to approach it**
Stack the two features into a matrix X of shape (30, 2). The weight vector w
starts at [0, 0]. The prediction for all rows at once is `X @ w + b` — a
(30,) vector. The gradient for every weight at once is `X.T @ err / m`, also
a vector. That is the whole change from one feature to two. First, run it on
the raw X with lr=0.001 and watch the loss explode to nan/inf — x2 is ~1000 so
its gradient is enormous. Then try a tiny lr (1e-6): no explosion, but w1
barely moves (stalled). Finally, standardise X with `StandardScaler`, run with
a large lr (0.5), and watch it converge in 100 epochs. To read the weights back
on the original scale, divide each scaled weight by the corresponding feature's
standard deviation.

**How the solution does it (in words)**
`gd_multi` is the vectorised version: `w` is a numpy array, `X @ w + b` gives
all predictions, and `X.T @ err / len(y)` updates both weights at once. On the
raw features, lr=0.001 diverges (nan/inf) because x2 ~ 1000 makes the w2
gradient huge; lr=1e-6 is safe but w1 stays at 0.017 instead of reaching the
true 1.8. After `StandardScaler`, lr=0.5 converges in 100 epochs to a loss of
0.74. The script then unscales the weights to recover w1 = 1.69 (true 1.8),
w2 = 0.0028 (true 0.004), and b = 4.37 (true 3.0) — close, proving the point:
scaling turned a narrow impossible valley into a round easy bowl.
