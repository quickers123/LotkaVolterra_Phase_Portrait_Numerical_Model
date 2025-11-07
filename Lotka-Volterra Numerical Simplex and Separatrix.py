import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

# Silence NumPy gradient edge warnings that appear when x-spacing becomes tiny.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# nice number formatter for terminal prints
def _fmt(x):  # 6 sig figs, switches to sci when needed
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)
    
def label_with_box(ax, x, y, text, dx=8, dy=8, ha="left", va="bottom", z=9):
    """Place a mathtext label in a rounded box offset from (x,y) by (dx,dy) points."""
    return ax.annotate(
        text,
        xy=(x, y), xycoords="data",
        xytext=(dx, dy), textcoords="offset points",  # offset in points
        ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.25", fc="0.92", ec="0.5", lw=0.8),
        zorder=z,
        clip_on=False,
    )    
    
    
# ===========================================================
# Competitive Lotka–Volterra model
#   dN1/dt = f(N1,N2) = N1 (3 - N1 - 1.5 N2)
#   dN2/dt = g(N1,N2) = N2 (4 - N2 - 2.5 N1)
# ===========================================================

def f(N1, N2):
    return N1 * (3.0 - N1 - 1.5 * N2)

def g(N1, N2):
    return N2 * (4.0 - N2 - 2.5 * N1)

def rhs(t, Y):
    N1, N2 = Y
    return np.array([f(N1, N2), g(N1, N2)])


# -----------------------------------------------------------
# Linearisation at the interior equilibrium and stable eigenvector
# -----------------------------------------------------------

def jacobian(N1, N2):
    df_dN1 = 3.0 - 2.0 * N1 - 1.5 * N2
    df_dN2 = -1.5 * N1
    dg_dN1 = -2.5 * N2
    dg_dN2 = 4.0 - 2.0 * N2 - 2.5 * N1
    return np.array([[df_dN1, df_dN2],
                     [dg_dN1, dg_dN2]])

# interior equilibrium (N1*, N2*) = (12/11, 14/11)
N1_star = 12.0 / 11.0
N2_star = 14.0 / 11.0

J_star = jacobian(N1_star, N2_star)
eigvals, eigvecs = np.linalg.eig(J_star)
stable_index = np.argmin(eigvals.real)  # the (most) negative eigenvalue
v_stable = eigvecs[:, stable_index].real
v_stable /= np.linalg.norm(v_stable)


# -----------------------------------------------------------
# Fourth–order Runge–Kutta stepper and integration utility
# -----------------------------------------------------------



def rk4_step_unit(Y, h):
    # unit-speed RHS
    V = np.array([f(Y[0], Y[1]), g(Y[0], Y[1])], dtype=float)
    sp = np.linalg.norm(V) + 1e-12
    F = lambda Z: np.array([f(Z[0], Z[1]), g(Z[0], Z[1])]) / (np.linalg.norm([f(Z[0], Z[1]), g(Z[0], Z[1])]) + 1e-12)

    k1 = F(Y)
    k2 = F(Y + 0.5*h*k1)
    k3 = F(Y + 0.5*h*k2)
    k4 = F(Y + h*k3)
    return Y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_until_axes_unit(Y0, h=5e-3, box_max=10.0, tol_axis=1e-6, max_steps=1_000_000):
    """
    Backward tracing along the *normalized* vector field (arc-length parameter).
    Use negative h to go "backward" (toward the axes from the saddle).
    """
    N1_vals = [Y0[0]]; N2_vals = [Y0[1]]
    Y = np.array(Y0, float)
    h = -abs(h)

    for _ in range(max_steps):
        Y = rk4_step_unit(Y, h)

        if (Y[0] < 0) or (Y[1] < 0) or (Y[0] > box_max) or (Y[1] > box_max):
            break

        N1_vals.append(Y[0]); N2_vals.append(Y[1])

        if (Y[1] < tol_axis and abs(Y[0] - 3.0) < 5e-2) or (Y[0] < tol_axis and abs(Y[1] - 4.0) < 5e-2):
            break

    return np.array(N1_vals), np.array(N2_vals)




def rk4_step(fun, t, Y, h):
    k1 = fun(t, Y)
    k2 = fun(t + 0.5 * h, Y + 0.5 * h * k1)
    k3 = fun(t + 0.5 * h, Y + 0.5 * h * k2)
    k4 = fun(t + h,       Y + h * k3)
    return Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def integrate_manifold(Y0, t0, t_end, h, box_max=6.0):
    if t_end == t0:
        return np.array([Y0[0]]), np.array([Y0[1]])

    n_steps = int(np.ceil(abs(t_end - t0) / abs(h)))
    h = np.sign(t_end - t0) * abs(h)

    N1_vals = []
    N2_vals = []

    t = t0
    Y = np.array(Y0, dtype=float)

    N1_vals.append(Y[0])
    N2_vals.append(Y[1])

    for _ in range(n_steps):
        Y = rk4_step(rhs, t, Y, h)
        t += h

        if (Y[0] < 0) or (Y[1] < 0) or (Y[0] > box_max) or (Y[1] > box_max):
            break

        N1_vals.append(Y[0])
        N2_vals.append(Y[1])

    return np.array(N1_vals), np.array(N2_vals)


# -----------------------------------------------------------
# Manifold tracing & simple polynomial least–squares fit (Script 1)
# -----------------------------------------------------------

epsilon = 1e-3
Y0_plus  = np.array([N1_star, N2_star]) + epsilon * v_stable
Y0_minus = np.array([N1_star, N2_star]) - epsilon * v_stable

# integrate backwards in time (stable manifold leaves the saddle as t -> -∞)
t0   = 0.0
tend = -20.0
h    = 1e-2

N1_plus,  N2_plus  = integrate_until_axes_unit(Y0_plus,  h=5e-3)
N1_minus, N2_minus = integrate_until_axes_unit(Y0_minus, h=5e-3)

# Build one continuous parametric curve (no sorting, no dx-based dedup)
# Reverse one branch so we go from near (0,4) → saddle → near (3,0)
N1_true = np.concatenate((N1_minus[::-1], [N1_star], N1_plus))
N2_true = np.concatenate((N2_minus[::-1], [N2_star], N2_plus))

# Keep only the biologically sensible region
mask_true = (N1_true >= 0.0) & (N2_true >= 0.0)
N1_true = N1_true[mask_true]
N2_true = N2_true[mask_true]


# --- Smooth prepend to (0,0) using the local asymptotic N2 ~ C * N1^(4/3) ---
if N1_true.size >= 2:
    x0 = float(N1_true[0]); y0 = float(N2_true[0])
    if x0 > 0 and y0 >= 0:
        C = y0 / (x0 ** (4.0/3.0))
        # create a short monotone segment back to the origin
        xs_pre = np.linspace(0.0, x0, 80, endpoint=False)  # exclude x0 to avoid duplicate
        ys_pre = C * (xs_pre ** (4.0/3.0))
        # keep strictly positive y (numerical safety)
        mask_pre = ys_pre >= 0.0
        N1_true = np.concatenate((xs_pre[mask_pre], N1_true))
        N2_true = np.concatenate((ys_pre[mask_pre], N2_true))



# Diagnostics for the "true" separatrix (finite-difference checks)
def separatrix_diagnostics(N1s, N2s):
    """Stable diagnostics on y(x): de-duplicate x, use 1st-order gradients,
    and guard all divisions."""
    if N1s.size < 5:
        return dict(rms=np.nan, min_slope=np.nan, min_curv=np.nan, n=len(N1s))

    # ensure strictly increasing x to avoid zero dx in gradient
    x = np.asarray(N1s, float)
    y = np.asarray(N2s, float)
    keep = np.r_[True, np.diff(x) > 1e-9]
    x = x[keep]; y = y[keep]

    # finite-only
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]

    if x.size < 5:
        return dict(rms=np.nan, min_slope=np.nan, min_curv=np.nan, n=len(x))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dydx   = np.gradient(y, x, edge_order=1)
        d2ydx2 = np.gradient(dydx, x, edge_order=1)

        fv = f(x, y)
        gv = g(x, y)
        safe = (np.abs(fv) > 1e-12) & np.isfinite(fv) & np.isfinite(gv) & np.isfinite(dydx)

        R = np.zeros_like(dydx)
        R[~safe] = np.nan
        R[safe] = dydx[safe] - gv[safe] / fv[safe]

        rms = float(np.sqrt(np.nanmean(R**2)))

    return dict(
        rms=rms,
        min_slope=float(np.nanmin(dydx)),
        min_curv=float(np.nanmin(d2ydx2)),
        n=int(x.size),
        xmin=float(x.min()), xmax=float(x.max()),
        ymin=float(y.min()), ymax=float(y.max())
    )


diag_true = separatrix_diagnostics(N1_true, N2_true)

# Print statement for the "true" separatrix, analogous to the simplex fit
print("\n================  NUMERICAL SEPARATRIX  ================")
print(f"Samples kept                : {_fmt(diag_true['n'])}")
print(f"x-range (N1)                : [{_fmt(diag_true['xmin'])}, {_fmt(diag_true['xmax'])}]")
print(f"y-range (N2)                : [{_fmt(diag_true['ymin'])}, {_fmt(diag_true['ymax'])}]")
print(f"RMS invariance residual     : {_fmt(diag_true['rms'])}   (R = y'(x) - g/f)")
print(f"min slope  y'(x)            : {_fmt(diag_true['min_slope'])}")
print(f"min curvature  y''(x)       : {_fmt(diag_true['min_curv'])}")

# Data for simple polynomial fit N2 ≈ p(N1)
# --- LSQ polynomial p(x) anchored at the origin to avoid negatives/kink ---
mask_fit = (N1_true > 0) & (N2_true >= 0)
x_fit = N1_true[mask_fit]
y_fit = N2_true[mask_fit]

# include the anchor (0,0)
x_fit_aug = np.r_[0.0, x_fit]
y_fit_aug = np.r_[0.0, y_fit]

poly_degree = 9  # keep moderate to avoid wiggles
coeffs_simple_hi2lo = np.polyfit(x_fit_aug, y_fit_aug, deg=poly_degree)  # high->low
poly_simple = np.poly1d(coeffs_simple_hi2lo)

# plot domain for this polynomial
N1_grid_simple = np.linspace(0.0, 3.5, 600)  # match ax.set_xlim(0, 3.5)
N2_poly_simple = poly_simple(N1_grid_simple)

# quality metric vs numerical separatrix (on the plotting domain)
mask_cmp = (N1_true >= N1_grid_simple.min()) & (N1_true <= N1_grid_simple.max())
rmse_sepx = np.nan
if np.count_nonzero(mask_cmp) > 5:
    y_hat = poly_simple(N1_true[mask_cmp])
    rmse_sepx = float(np.sqrt(np.mean((y_hat - N2_true[mask_cmp])**2)))

# coefficients in power basis low->high: b[k] for x^k
b_power = poly_simple.c[::-1]

print("\n================  SEPARATRIX (LSQ polynomial)  =================")
print(f"Polynomial degree           : {poly_degree}")
print(f"Fit sample size             : {_fmt(x_fit_aug.size)} (incl. origin)")
print(f"Plot x-domain               : [0, {_fmt(N1_grid_simple.max())}]")
print(f"RMSE vs numerical separatrix: {_fmt(rmse_sepx)}")

print("\n----------------  Power-basis coefficients (separatrix)  ----------------")
print("p(x) = Σ_{k=0}^{deg} b[k] x^k")
for k, bk in enumerate(b_power):
    print(f"b[{k:>2}] = {_fmt(bk):>16}")

desmos_sep = "y=" + "+".join(
    (f"{b_power[0]:.12g}",) +
    tuple(f"{b_power[k]:.12g}*x^{k}" for k in range(1, len(b_power)))
)
print("\nDesmos one-liner (separatrix):")
print(desmos_sep)


# -----------------------------------------------------------
# Anchor-constrained polynomial optimisation fit (Script 2)
# -----------------------------------------------------------

# Anchors on the carrying simplex
x0, y0 = 0.0, 4.0
xs, ys = 12.0/11.0, 14.0/11.0
x3, y3 = 3.0, 0.0

def anchor_polys():
    d0 = (x0 - xs)*(x0 - x3)
    ds = (xs - x0)*(xs - x3)
    d3 = (x3 - x0)*(x3 - xs)

    L0 = np.poly1d([1.0, -(xs + x3), xs*x3]) / d0
    Ls = np.poly1d([1.0, -(x0 + x3), x0*x3]) / ds
    L3 = np.poly1d([1.0, -(x0 + xs), x0*xs]) / d3

    yA  = y0*L0 + ys*Ls + y3*L3
    yA1 = np.polyder(yA, 1)
    yA2 = np.polyder(yA, 2)

    A = np.poly1d([1.0, -(x0 + xs + x3),
                   (x0*xs + x0*x3 + xs*x3),
                   -x0*xs*x3])
    A1 = np.polyder(A, 1)
    A2 = np.polyder(A, 2)
    return yA, yA1, yA2, A, A1, A2

yA, yA1, yA2, A, A1, A2 = anchor_polys()

def build_poly_and_derivs(coeff_free):
    q = np.poly1d(list(coeff_free)[::-1])  # power-basis coeffs -> highest-first
    q1 = np.polyder(q, 1)
    q2 = np.polyder(q, 2)

    def y(x):
        x = np.asarray(x)
        return np.polyval(yA, x) + np.polyval(A, x)*np.polyval(q, x)

    def yp(x):
        x = np.asarray(x)
        return (np.polyval(yA1, x)
                + np.polyval(A1, x)*np.polyval(q, x)
                + np.polyval(A,  x)*np.polyval(q1, x))

    def ypp(x):
        x = np.asarray(x)
        return (np.polyval(yA2, x)
                + np.polyval(A2, x)*np.polyval(q,  x)
                + 2*np.polyval(A1, x)*np.polyval(q1, x)
                +   np.polyval(A,  x)*np.polyval(q2, x))
    return y, yp, ypp

def fit_carrying_simplex_poly(deg=8, n_grid=500, lam_shape=120.0, lam_smooth=1e-7, seed=2):
    rng = np.random.default_rng(seed)
    if deg < 3:
        raise ValueError("Degree must be at least 3 to satisfy three anchors.")
    deg_q = deg - 3

    # collocation grid (avoid exact anchors slightly)
    xsamp = np.linspace(0.0, 3.0, n_grid)
    xsamp = xsamp[(xsamp > 1e-9) & (xsamp < 3.0-1e-9)]
    xsamp += 1e-6 * rng.standard_normal(xsamp.size)

    c0 = np.zeros(deg_q + 1)

    def objective(cfree):
        y, yp, ypp = build_poly_and_derivs(cfree)
        xi = xsamp
        yi  = y(xi)
        fi  = f(xi, yi)
        gi  = g(xi, yi)
        R = gi - yp(xi)*fi  # invariance residual
        J_inv = np.sum(R*R)

        # shape penalties (monotone decreasing & convex)
        ypi  = yp(xi)
        yppi = ypp(xi)
        pen_mon  = np.sum(np.maximum(0.0, ypi)**2)     # enforce y' ≤ 0
        pen_conv = np.sum(np.maximum(0.0, -yppi)**2)   # enforce y'' ≥ 0

        reg = np.sum(cfree*cfree)
        return J_inv + lam_shape*(pen_mon + pen_conv) + lam_smooth*reg

    res = minimize(objective, c0, method="L-BFGS-B",
                   options=dict(maxiter=3000, ftol=1e-12, gtol=1e-10))
    copt = res.x
    y, yp, ypp = build_poly_and_derivs(copt)

    # recover full power-basis coefficients a_k for y(x) by least squares on a fine grid
    xfine = np.linspace(0.0, 3.0, 1201)
    V = np.column_stack([xfine**k for k in range(deg+1)])
    a, *_ = np.linalg.lstsq(V, y(xfine), rcond=None)

    # diagnostics
    xi = np.linspace(0,3,601)
    print("\n================  CARRYING SIMPLEX (poly fit)  ================")
    print(f"Polynomial degree           : {deg}")
    print(f"Anchors                     : y(0)={_fmt(y(0.0))}, y(xs)={_fmt(y(xs))}, y(3)={_fmt(y(3.0))}")

    xi = np.linspace(0, 3, 601)
    yp_min  = float(np.min(yp(xi)))
    ypp_min = float(np.min(ypp(xi)))
    R = g(xi, y(xi)) - yp(xi)*f(xi, y(xi))
    rms_inv = float(np.sqrt(np.mean(R*R)))

    viol_mon  = int(np.count_nonzero(yp(xi) > 0))   # should be ≤ 0
    viol_conv = int(np.count_nonzero(ypp(xi) < 0))  # should be ≥ 0

    print(f"min slope  y'(x)            : {_fmt(yp_min)}   (monotonic ↓ violations: {viol_mon})")
    print(f"min curvature  y''(x)       : {_fmt(ypp_min)}  (convexity violations: {viol_conv})")
    print(f"RMS invariance residual     : {_fmt(rms_inv)}")

    return a, y, yp, ypp

# -----------------------------------------------------------
# Run & combined plot
# -----------------------------------------------------------

if __name__ == "__main__":
    # Anchor-constrained polynomial (deg 8 by default)
    DEG = 8
    a_power, y_opt, yp_opt, ypp_opt = fit_carrying_simplex_poly(
        deg=DEG, n_grid=500, lam_shape=120.0, lam_smooth=1e-7, seed=2
    )

    # Print power-basis coefficients (useful for Desmos)
    print("\n----------------  Power-basis coefficients (carrying simplex)  ----------------")
    print("y(x) = Σ_{k=0}^{deg} a[k] x^k")
    for k, ak in enumerate(a_power):
        print(f"a[{k:>2}] = {_fmt(ak):>16}")

    desmos = "y=" + "+".join(
        (f"{a_power[0]:.12g}",) + tuple(f"{a_power[k]:.12g}*x^{k}" for k in range(1, len(a_power)))
    )
    print("\nDesmos one-liner (carrying simplex):")
    print(desmos)

    # -------------------------------------------------------
    # Plotting: vector field, nullclines, numerical separatrix (single curve),
    #           simple poly (deg4), and anchor-constrained poly (deg8)
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # Vector field (quiver)
    N1_vals_plot = np.linspace(0, 3.5, 20)
    N2_vals_plot = np.linspace(0, 4.5, 20)
    N1_grid_q, N2_grid_q = np.meshgrid(N1_vals_plot, N2_vals_plot)
    U = f(N1_grid_q, N2_grid_q)
    V = g(N1_grid_q, N2_grid_q)
    speed = np.sqrt(U**2 + V**2)
    U_plot = U / (speed + 1e-9)
    V_plot = V / (speed + 1e-9)
    ax.quiver(N1_grid_q, N2_grid_q, U_plot, V_plot,
              angles="xy", scale_units="xy", scale=20, alpha=0.35, zorder=1)

    # Nullclines
    ax.plot(N1_vals_plot, (3.0 - N1_vals_plot) / 1.5, "g--", lw=1.5, label=r"$N_1$ nullcline", zorder=2)
    ax.plot(N1_vals_plot, 4.0 - 2.5 * N1_vals_plot,   "r--", lw=1.5, label=r"$N_2$ nullcline", zorder=2)
    # --- Region shading around the saddle using separatrix (SP) and carrying simplex (CS) ---
    # Define a fine x-grid on the model domain (CS is defined on [0,3])
    x_shade = np.linspace(0.0, 5.0, 1200)
    cs_vals = y_opt(x_shade)               # Carrying simplex CS(x)
    sp_vals = poly_simple(x_shade)         # Separatrix SP(x)

    # Clip to plotting y-range and guard against NaNs/Infs
    y_lo, y_hi = 0.0, 4.5
    cs_vals = np.clip(cs_vals, y_lo, y_hi)
    sp_vals = np.clip(sp_vals, y_lo, y_hi)
    finite = np.isfinite(cs_vals) & np.isfinite(sp_vals)

    # Colors (soft/opaque enough to be visible but not dominant)
    col_green  = (0.10, 0.65, 0.10, 0.12)   # RGBA
    col_purple = (0.50, 0.20, 0.70, 0.12)
    col_blue   = (0.15, 0.35, 0.85, 0.12)
    col_red    = (0.85, 0.20, 0.20, 0.12)

    # Helper arrays
    top_band    = np.full_like(x_shade, y_hi)
    bottom_band = np.full_like(x_shade, y_lo)

    # Green: basin of (0,4) AND above CS  ⇒ y ∈ [max(CS,SP), y_hi]
    ax.fill_between(
        x_shade, np.maximum(cs_vals, sp_vals), top_band,
        where=finite, color=col_green, step=None, linewidth=0, zorder=0
    )

    # Blue: basin of (3,0) AND below CS  ⇒ y ∈ [y_lo, min(CS,SP)]
    ax.fill_between(
        x_shade, bottom_band, np.minimum(cs_vals, sp_vals),
        where=finite, color=col_blue, step=None, linewidth=0, zorder=0
    )

    # Purple: basin of (0,4) AND below CS (i.e., SP < y < CS when CS > SP)
    mask_purple = finite & (cs_vals > sp_vals)
    ax.fill_between(
        x_shade, sp_vals, cs_vals,
        where=mask_purple, color=col_purple, step=None, linewidth=0, zorder=0
    )

    # Red: basin of (3,0) AND above CS (i.e., CS < y < SP when SP > CS)
    mask_red = finite & (sp_vals > cs_vals)
    ax.fill_between(
        x_shade, cs_vals, sp_vals,
        where=mask_red, color=col_red, step=None, linewidth=0, zorder=0
    )


    # --- Phase trajectories from specified initial points ---
    seeds = [
        (1,1),(1,2),(1,3),
        (2,1),(2,2),(2,3),
        (3,1),(3,2),(3,3),
        (0.5,1),(0.5,2),(0.5,3),(0.5,0.5),
        (1,0.5),(2,0.5),(3,0.5),
        (0.5,4),(1,4),(2,4),(3,4),
        (0.25,3),(0.25,2),(0.25,1),(0.25,0.5),(0.25,4),
        (0.25,0.25),(0.5,0.25),(1,0.25),(2,0.25),(3,0.25),
        (3.25,0.25),(3.25,0.5),(3.25,1),(3.25,2),(3.25,3),(3.25,4)
    ]

    t_span = 40.0          # integrate +/- this amount of time
    h_traj = 1e-2          # time step for RK4
    box_bound = 6.0        # keep inside biologically sensible box

    added_label = False
    for (x0, y0) in seeds:
        # forward in time
        N1_f, N2_f = integrate_manifold([x0, y0], t0=0.0, t_end= t_span, h=h_traj, box_max=box_bound)
        # backward in time
        N1_b, N2_b = integrate_manifold([x0, y0], t0=0.0, t_end=-t_span, h=h_traj, box_max=box_bound)

        # stitch into one trajectory
        N1_traj = np.concatenate((N1_b[::-1], N1_f))
        N2_traj = np.concatenate((N2_b[::-1], N2_f))

        ax.plot(
            N1_traj, N2_traj,
            color="0.35", lw=1.6, alpha=0.9, zorder=3,
            label="phase trajectories" if not added_label else None
        )
        added_label = True
    # mark initial points
    ax.scatter(
        [s[0] for s in seeds], [s[1] for s in seeds],
        s=18, facecolors="none", edgecolors="0.2", linewidths=1.2, zorder=7,
        label="initial points"
    )





    # Numerical separatrix as ONE curve (merged branches)
    ax.plot(N1_true, N2_true, color="tab:blue", lw=2.2, label="numerical separatrix", zorder=4)

    # Simple polynomial LSQ (deg 4)
    ax.plot(N1_grid_simple, N2_poly_simple, "k-", lw=2.0, label=f"P({poly_degree}) LSQ Seperatrix", zorder=5)

    # Anchor-constrained optimisation polynomial (deg 8)
    xs_plot = np.linspace(0, 3, 800)
    ax.plot(xs_plot, y_opt(xs_plot), color="#FF7F0E", lw=2.4,
            label=f"P({DEG}) LSQ Carrying Simplex", zorder=6)

    # Equilibria
    ax.plot([0, 3, 0, N1_star],
            [0, 0, 4, N2_star],
            "ko", ms=5, zorder=7)



    label_with_box(ax, 3, 0,  r"$(3,0) = n_{2}^{*}$",  dx=8,  dy=8,  ha="left",  va="bottom")
    label_with_box(ax, 0, 4,  r"$(0,4) = n_{3}^{*}$",  dx=8,  dy=10, ha="left",  va="bottom")
    label_with_box(ax, N1_star, N2_star,
                r"$(\frac{12}{11},\,\frac{14}{11}) = n_{4}^{*}$",
                dx=10, dy=10, ha="left", va="bottom")
    label_with_box(ax, 0, 0,  r"$(0,0) = n_{1}^{*}$",  dx=8,  dy=8,  ha="left",  va="bottom")

    
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel(r"$N_1$")
    ax.set_ylabel(r"$N_2$")
    ax.set_title("Lotka Voleterra Phase Plane with Numerical LSQ fits of Separatrix & Carrying Simplex Curves.")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
