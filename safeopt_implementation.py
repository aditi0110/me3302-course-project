"""
SafeOpt — 2-Parameter Vial Filling Optimiser
=============================================
Parameters tuned:
    h  — nozzle height   (mm)
    v  — fluid velocity  (mL/s)

Assumption (WLOG):
    Bubble count follows a convex quadratic bowl over (h, v).
    True minimum at (h_opt, v_opt) — unknown to the algorithm.
    SafeOpt explores safely from a known seed and should converge
    toward the global minimum.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import GPy
import safeopt

warnings.filterwarnings("ignore")


PARAM_BOUNDS = {
    "nozzle_height_mm":   (5.0, 25.0),   # mm
    "fluid_velocity_mLs": (0.5,  5.0),   # mL/s
}

GRID_RESOLUTION = {
    "nozzle_height_mm":   30,
    "fluid_velocity_mLs": 30,
}


INITIAL_SAFE_PARAMS = {
    "nozzle_height_mm":   15.0,
    "fluid_velocity_mLs":  2.0,
}
INITIAL_BUBBLE_COUNT = 6    
H_OPT = 15.0   # mm
V_OPT =  2.5   # mL/s

SAFETY_THRESHOLD   = 10     # bubbles — fills above this are unsafe
NOISE_STD          = 0.4    # observation noise (bubbles)
LENGTHSCALES       = [3.5, 0.7]   # RBF lengthscales: ~15-20% of each range
BETA               = 2.0    # confidence-interval scaling (≈95 %)
OPTIMIZE_GP        = True   # re-optimise GP hyperparams each iteration
MAX_ITERATIONS     = 25

OUTPUT_DIR = "safeopt_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)



SAFETY_REWARD_THRESHOLD = 0.0

def bubbles_to_reward(b: float) -> float:
    return float(SAFETY_THRESHOLD - b)

def reward_to_bubbles(r: float) -> float:
    return float(SAFETY_THRESHOLD - r)



param_names = list(PARAM_BOUNDS.keys())

def build_grid(bounds, resolution):
    axes = [np.linspace(bounds[k][0], bounds[k][1], resolution[k])
            for k in bounds]
    H_ax, V_ax = axes
    HH, VV = np.meshgrid(H_ax, V_ax, indexing="ij")
    grid = np.column_stack([HH.ravel(), VV.ravel()])
    return grid, H_ax, V_ax

parameter_grid, H_axis, V_axis = build_grid(PARAM_BOUNDS, GRID_RESOLUTION)
print(f"Grid: {len(parameter_grid):,} points  "
      f"({GRID_RESOLUTION['nozzle_height_mm']} h × "
      f"{GRID_RESOLUTION['fluid_velocity_mLs']} v)")



# bubble_count(h, v) = c_h*(h−h_opt)² + c_v*(v−v_opt)² + noise
#
# Coefficients chosen so that:
#   • minimum value ≈ 0 bubbles at (H_OPT, V_OPT)
#   • maximum at grid corners ≈ 20 bubbles  (well inside [0, 50])
#   • fluid velocity has stronger curvature (more sensitive)

C_H = 0.06    # curvature along nozzle height
C_V = 1.20    # curvature along fluid velocity  (higher → more sensitive)

def true_bubble_count(h: float, v: float) -> float:
    """Noiseless ground truth."""
    return C_H * (h - H_OPT)**2 + C_V * (v - V_OPT)**2

def simulated_bubble_count(params: dict) -> float:
    """Noisy observation returned to SafeOpt."""
    h = params["nozzle_height_mm"]
    v = params["fluid_velocity_mLs"]
    b = true_bubble_count(h, v) + np.random.normal(0, NOISE_STD)
    return float(np.clip(b, 0, 50))

_H_dense = np.linspace(PARAM_BOUNDS["nozzle_height_mm"][0],
                        PARAM_BOUNDS["nozzle_height_mm"][1], 200)
_V_dense = np.linspace(PARAM_BOUNDS["fluid_velocity_mLs"][0],
                        PARAM_BOUNDS["fluid_velocity_mLs"][1], 200)
HH_dense, VV_dense = np.meshgrid(_H_dense, _V_dense, indexing="ij")
BB_dense = true_bubble_count(HH_dense, VV_dense)          # (200,200)





def params_to_array(p: dict) -> np.ndarray:
    return np.array([[p[k] for k in param_names]])

def array_to_params(arr: np.ndarray) -> dict:
    row = arr.flatten()
    return {k: float(row[i]) for i, k in enumerate(param_names)}

def snap_to_grid(params: dict) -> np.ndarray:
    target = params_to_array(params)
    ranges = np.array([PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0]
                       for k in param_names])
    dists  = np.linalg.norm((parameter_grid - target) / ranges, axis=1)
    idx    = np.argmin(dists)
    return parameter_grid[idx:idx+1, :]



x0       = snap_to_grid(INITIAL_SAFE_PARAMS)
y0_reward = np.array([[bubbles_to_reward(INITIAL_BUBBLE_COUNT)]])

print(f"\nSeed point : h={x0[0,0]:.2f} mm,  v={x0[0,1]:.3f} mL/s")
print(f"Seed reward: {y0_reward[0,0]:.2f}  (bubble count = {INITIAL_BUBBLE_COUNT})")

kernel = GPy.kern.RBF(
    input_dim=2,
    variance=1.0,
    lengthscale=LENGTHSCALES,
    ARD=True,
)

gp = GPy.models.GPRegression(x0, y0_reward, kernel,
                              noise_var=NOISE_STD**2)
gp.Gaussian_noise.variance.fix()   # keep noise fixed at our prior estimate

if OPTIMIZE_GP:
    gp.optimize(messages=False)

print("GP initialised.\n")



opt = safeopt.SafeOpt(
    gp,
    parameter_grid,
    fmin=[SAFETY_REWARD_THRESHOLD],
    beta=BETA,
    threshold=0.0,
)



CLR_SAFE_FILL  = "#d4edda"    # light green — safe set
CLR_UNSAFE     = "#f8d7da"    # light red   — unsafe region
CLR_TRAJ       = "#1f77b4"    # blue        — trajectory
CLR_EVAL       = "#ff7f0e"    # orange      — evaluated points
CLR_BEST       = "#d62728"    # red         — current best
CLR_OPT        = "#2ca02c"    # green star  — true optimum
CLR_CONTOUR    = "white"      # contour labels
CONTOUR_LEVELS = [0, 1, 2, 4, 6, 8, 10, 14, 18]

def plot_iteration(iteration: int,
                   eval_h: list, eval_v: list,
                   bubble_counts: list,
                   safe_mask: np.ndarray,
                   x_next_h: float, x_next_v: float):

    fig, ax = plt.subplots(figsize=(7, 5.5))

    cf = ax.contourf(HH_dense, VV_dense, BB_dense,
                     levels=CONTOUR_LEVELS,
                     cmap="YlOrRd_r", alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("True bubble count", fontsize=9)

    ax.contour(HH_dense, VV_dense, BB_dense,
               levels=[SAFETY_THRESHOLD],
               colors=["#cc0000"], linewidths=[2.0],
               linestyles=["--"])
    ax.text(6.5, 4.55,
            f"Safety limit\n({SAFETY_THRESHOLD} bubbles)",
            color="#cc0000", fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    if safe_mask is not None and safe_mask.sum() > 0:
        safe_pts = parameter_grid[safe_mask]
        ax.scatter(safe_pts[:, 0], safe_pts[:, 1],
                   s=6, c=CLR_SAFE_FILL, marker="s",
                   alpha=0.35, linewidths=0,
                   label=f"GP safe set ({safe_mask.sum()} pts)",
                   zorder=2)

    for i in range(1, len(eval_h)):
        ax.annotate(
            "", xy=(eval_h[i], eval_v[i]),
            xytext=(eval_h[i-1], eval_v[i-1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=CLR_TRAJ,
                lw=1.4,
                mutation_scale=12,
                connectionstyle="arc3,rad=0.05",
            ),
            zorder=4,
        )

    best_idx = int(np.argmin(bubble_counts))
    for i, (h, v, b) in enumerate(zip(eval_h, eval_v, bubble_counts)):
        is_best    = (i == best_idx)
        is_current = (i == len(eval_h) - 1)
        color  = CLR_BEST  if is_best    else CLR_EVAL
        marker = "*"       if is_best    else "o"
        size   = 220       if is_best    else 90
        zorder = 7         if is_best    else 5

        ax.scatter(h, v, s=size, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8,
                   zorder=zorder)

        # circle the newest point
        if is_current and not is_best:
            ax.scatter(h, v, s=260, facecolors="none",
                       edgecolors=CLR_TRAJ, linewidths=2.0,
                       zorder=6)

        # iteration label
        label_text = f"{i}"
        ax.text(h + 0.28, v + 0.08, label_text,
                fontsize=7.5, color="white", fontweight="bold",
                zorder=8,
                path_effects=[
                    pe.withStroke(linewidth=2, foreground="black")
                ])

    ax.scatter(H_OPT, V_OPT, s=260, c=CLR_OPT,
               marker="*", edgecolors="white",
               linewidths=0.8, zorder=9,
               label=f"True optimum ({H_OPT}, {V_OPT})")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLR_EVAL, markersize=8,
               label="Evaluated point"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=CLR_BEST, markersize=10,
               label="Current best"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=CLR_OPT, markersize=12,
               label=f"True optimum"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=CLR_SAFE_FILL,
               markeredgecolor="gray",
               markersize=8, label="GP safe set"),
        Line2D([0], [0], color="#cc0000", lw=2,
               linestyle="--", label="Safety boundary"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=7.5, framealpha=0.85)

    ax.set_xlabel("Nozzle height  $h$  (mm)", fontsize=10)
    ax.set_ylabel("Fluid velocity  $v$  (mL/s)", fontsize=10)
    ax.set_xlim(PARAM_BOUNDS["nozzle_height_mm"])
    ax.set_ylim(PARAM_BOUNDS["fluid_velocity_mLs"])

    best_b = bubble_counts[best_idx]
    ax.set_title(
        f"SafeOpt — Iteration {iteration}\n"
        f"Evaluated point:  h={x_next_h:.2f} mm,  "
        f"v={x_next_v:.3f} mL/s    "
        f"(bubbles = {bubble_counts[-1]:.2f})\n"
        f"Best so far: {best_b:.2f} bubbles  "
        f"@ h={eval_h[best_idx]:.2f}, v={eval_v[best_idx]:.3f}",
        fontsize=9, loc="left",
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"iteration_{iteration:03d}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved -> {path}")


history = {
    "iteration":     [0],
    "h":             [float(x0[0, 0])],
    "v":             [float(x0[0, 1])],
    "bubble_count":  [float(INITIAL_BUBBLE_COUNT)],
    "reward":        [float(y0_reward[0, 0])],
    "safe_set_size": [],
}

best_bubble_count = INITIAL_BUBBLE_COUNT
best_h = float(x0[0, 0])
best_v = float(x0[0, 1])

# Draw iteration-0 state (seed only, no trajectory yet)
plot_iteration(
    iteration=0,
    eval_h=[float(x0[0, 0])],
    eval_v=[float(x0[0, 1])],
    bubble_counts=[float(INITIAL_BUBBLE_COUNT)],
    safe_mask=opt.S,
    x_next_h=float(x0[0, 0]),
    x_next_v=float(x0[0, 1]),
)


print("=" * 60)
print("Starting SafeOpt iterations …")
print("=" * 60)

for iteration in range(1, MAX_ITERATIONS + 1):

    try:
        x_next = opt.optimize()
    except Exception as exc:
        print(f"\nSafeOpt error at iteration {iteration}: {exc}")
        break

    if x_next is None:
        print("\nSafeOpt: safe set exhausted or fully converged.")
        break

    next_h = float(x_next.flatten()[0])
    next_v = float(x_next.flatten()[1])

    print(f"\n{'-'*60}")
    print(f"Iteration {iteration:>3d}  ->  "
          f"h = {next_h:6.2f} mm,  v = {next_v:5.3f} mL/s")

    bubble_count = simulated_bubble_count(
        {"nozzle_height_mm": next_h, "fluid_velocity_mLs": next_v}
    )
    reward = bubbles_to_reward(bubble_count)
    safe_str = "SAFE" if reward >= SAFETY_REWARD_THRESHOLD else "UNSAFE ⚠"
    print(f"           bubbles = {bubble_count:.2f}   "
          f"reward = {reward:.2f}   [{safe_str}]")

    opt.add_new_data_point(x_next, np.array([[reward]]))

    if OPTIMIZE_GP:
        try:
            gp.optimize(messages=False)
        except Exception:
            pass

    safe_set_size = int(np.sum(opt.S))
    history["iteration"].append(iteration)
    history["h"].append(next_h)
    history["v"].append(next_v)
    history["bubble_count"].append(bubble_count)
    history["reward"].append(reward)
    history["safe_set_size"].append(safe_set_size)

    if bubble_count < best_bubble_count:
        best_bubble_count = bubble_count
        best_h, best_v = next_h, next_v
        print(f"New best!  {bubble_count:.2f} bubbles "
              f"@ h={next_h:.2f}, v={next_v:.3f}")

    print(f"  Safe set: {safe_set_size:,} / {len(parameter_grid):,} grid points")

    plot_iteration(
        iteration=iteration,
        eval_h=history["h"],
        eval_v=history["v"],
        bubble_counts=history["bubble_count"],
        safe_mask=opt.S,
        x_next_h=next_h,
        x_next_v=next_v,
    )


n_evals = len(history["iteration"])
print("\n" + "=" * 60)
print("OPTIMISATION COMPLETE")
print("=" * 60)
print(f"\nTotal evaluations : {n_evals}")
print(f"Best bubble count : {best_bubble_count:.2f}")
print(f"Best parameters   : h = {best_h:.2f} mm,  v = {best_v:.3f} mL/s")
print(f"True optimum      : h = {H_OPT:.2f} mm,  v = {V_OPT:.3f} mL/s")
print(f"True minimum      : {true_bubble_count(H_OPT, V_OPT):.2f} bubbles")
dist = np.sqrt((best_h - H_OPT)**2 + (best_v - V_OPT)**2)
print(f"Distance to optimum (Euclidean, unnorm.) : {dist:.3f}")

print(f"\n{'Iter':>5}  {'h (mm)':>8}  {'v (mL/s)':>10}  "
      f"{'Bubbles':>8}  {'Reward':>8}")
for idx in range(n_evals):
    print(f"  {history['iteration'][idx]:>3}  "
          f"  {history['h'][idx]:>7.2f}"
          f"  {history['v'][idx]:>10.3f}"
          f"  {history['bubble_count'][idx]:>8.2f}"
          f"  {history['reward'][idx]:>8.2f}")



iters   = history["iteration"]
bubbles = history["bubble_count"]
running_best = list(np.minimum.accumulate(bubbles))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("SafeOpt — Vial Filling (2-Parameter)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(iters, bubbles, "o-", color="steelblue",
        markersize=5, label="Observed")
ax.plot(iters, running_best, "--", color="tomato",
        linewidth=2, label="Running best")
ax.axhline(SAFETY_THRESHOLD, color="orange", linestyle=":",
           linewidth=1.5, label=f"Safety limit ({SAFETY_THRESHOLD})")
ax.axhline(true_bubble_count(H_OPT, V_OPT), color="green",
           linestyle=":", linewidth=1.5, label="True minimum (0)")
ax.set_xlabel("Iteration"); ax.set_ylabel("Bubble count")
ax.set_title("Bubble Count over Iterations")
ax.legend(fontsize=7)
ax.set_ylim(bottom=-0.5)

ax = axes[1]
if history["safe_set_size"]:
    ax.plot(iters[1:], history["safe_set_size"],
            "s-", color="seagreen", markersize=5)
ax.axhline(len(parameter_grid), color="gray", linestyle="--",
           linewidth=1, label="Total grid points")
ax.set_xlabel("Iteration"); ax.set_ylabel("Safe set size")
ax.set_title("Safe Set Growth")
ax.legend(fontsize=7)

ax = axes[2]
cf = ax.contourf(HH_dense, VV_dense, BB_dense,
                 levels=CONTOUR_LEVELS, cmap="YlOrRd_r", alpha=0.8)
fig.colorbar(cf, ax=ax, pad=0.02).set_label("True bubbles", fontsize=8)
ax.contour(HH_dense, VV_dense, BB_dense,
           levels=[SAFETY_THRESHOLD], colors=["#cc0000"],
           linewidths=[1.8], linestyles=["--"])

h_vals = history["h"]
v_vals = history["v"]
sc = ax.scatter(h_vals, v_vals,
                c=bubbles, cmap="coolwarm_r",
                s=60, edgecolors="white", linewidths=0.6,
                vmin=0, vmax=SAFETY_THRESHOLD, zorder=5)
fig.colorbar(sc, ax=ax, pad=0.02).set_label("Observed bubbles", fontsize=8)
ax.plot(h_vals, v_vals, "-", color="steelblue",
        alpha=0.5, linewidth=1.2, zorder=4)
ax.scatter(H_OPT, V_OPT, s=250, c="lime", marker="*",
           edgecolors="black", linewidths=0.8, zorder=9,
           label="True optimum")
ax.scatter(best_h, best_v, s=200, c="red", marker="*",
           edgecolors="white", linewidths=0.8, zorder=8,
           label="Found best")
ax.set_xlabel("Nozzle height $h$ (mm)")
ax.set_ylabel("Fluid velocity $v$ (mL/s)")
ax.set_title("Full Evaluation Trajectory")
ax.set_xlim(PARAM_BOUNDS["nozzle_height_mm"])
ax.set_ylim(PARAM_BOUNDS["fluid_velocity_mLs"])
ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, "summary.png")
plt.savefig(summary_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nSummary plot saved -> {summary_path}")
print(f"All per-iteration plots saved in -> {OUTPUT_DIR}/")