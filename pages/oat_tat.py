"""
Page 1 — OAT / TAT Squeezing
All content from Final_5_18.py, unchanged.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.title("2D Lattice Spin Squeezing Simulator")
st.markdown(
    r"""
This app compares **one-axis twisting** and **generalized two-axis twisting**
on an \(n\times n\) spin-1/2 lattice using a stochastic DTWA / Monte Carlo method.

The total number of spins is \(N = n^2\).

Instead of evolving the full \(2^N\)-dimensional quantum state, the app samples
many stochastic spin trajectories and averages observables.
"""
)


# ============================================================
# Physics helpers
# ============================================================

def lattice_positions_2d(n, spacing=1.0):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.column_stack([xs.ravel(), ys.ravel()]) * spacing


def build_coupling_matrix(n, geometry, coupling_range, alpha, normalize=True):
    pos = lattice_positions_2d(n)
    N = n * n
    J = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = np.linalg.norm(pos[i] - pos[j])
            if geometry == "all_to_all":
                J[i, j] = 1.0
            elif geometry == "nearest":
                if np.isclose(r, 1.0):
                    J[i, j] = 1.0
            elif geometry == "finite_range":
                if r <= coupling_range:
                    J[i, j] = 1.0
            elif geometry == "power_law":
                if r <= coupling_range:
                    J[i, j] = 1.0 / (r ** alpha)
            else:
                raise ValueError("Unknown geometry.")
    if normalize:
        row_sums = np.sum(np.abs(J), axis=1)
        nonzero = row_sums[row_sums > 0]
        if len(nonzero) > 0:
            J = J / np.mean(nonzero)
    return J


def sample_initial_spins(num_samples, N, initial_axis="x", seed=None):
    rng = np.random.default_rng(seed)
    spins = np.zeros((num_samples, N, 3), dtype=float)
    s1 = rng.choice([-0.5, 0.5], size=(num_samples, N))
    s2 = rng.choice([-0.5, 0.5], size=(num_samples, N))
    if initial_axis == "x":
        spins[:, :, 0] = 0.5;  spins[:, :, 1] = s1;  spins[:, :, 2] = s2
    elif initial_axis == "y":
        spins[:, :, 0] = s1;   spins[:, :, 1] = 0.5; spins[:, :, 2] = s2
    elif initial_axis == "z":
        spins[:, :, 0] = s1;   spins[:, :, 1] = s2;  spins[:, :, 2] = 0.5
    else:
        raise ValueError("initial_axis must be x, y, or z.")
    return spins


def make_hamiltonian_couplings(model, n, chi, geometry, coupling_range, alpha, tat_jz_over_jy):
    base = build_coupling_matrix(n, geometry, coupling_range, alpha, normalize=True)
    N = n * n
    Jx = np.zeros((N, N)); Jy = np.zeros((N, N)); Jz = np.zeros((N, N))
    if model == "OAT":
        Jz = 2.0 * chi * base
    elif model == "TAT":
        Jy = -2.0 * chi * base
        Jz =  2.0 * chi * tat_jz_over_jy * base
    else:
        raise ValueError("model must be OAT or TAT.")
    return Jx, Jy, Jz


def spin_derivative(spins, Jx, Jy, Jz):
    Sx, Sy, Sz = spins[:, :, 0], spins[:, :, 1], spins[:, :, 2]
    Bx = Sx @ Jx.T;  By = Sy @ Jy.T;  Bz = Sz @ Jz.T
    dS = np.empty_like(spins)
    dS[:, :, 0] = By * Sz - Bz * Sy
    dS[:, :, 1] = Bz * Sx - Bx * Sz
    dS[:, :, 2] = Bx * Sy - By * Sx
    return dS


def rk4_step(spins, dt, Jx, Jy, Jz):
    k1 = spin_derivative(spins,              Jx, Jy, Jz)
    k2 = spin_derivative(spins + 0.5*dt*k1,  Jx, Jy, Jz)
    k3 = spin_derivative(spins + 0.5*dt*k2,  Jx, Jy, Jz)
    k4 = spin_derivative(spins +    dt*k3,   Jx, Jy, Jz)
    return spins + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def perpendicular_basis(mean_J):
    norm = np.linalg.norm(mean_J)
    if norm < 1e-12:
        return np.array([0., 1., 0.]), np.array([0., 0., 1.])
    e0 = mean_J / norm
    trial = np.array([0., 0., 1.])
    if abs(np.dot(e0, trial)) > 0.9:
        trial = np.array([0., 1., 0.])
    e1 = trial - np.dot(trial, e0) * e0
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def compute_observables(spins):
    num_samples, N, _ = spins.shape
    J_samples = np.sum(spins, axis=1)
    mean_J    = np.mean(J_samples, axis=0)
    mean_norm = np.linalg.norm(mean_J)
    e1, e2    = perpendicular_basis(mean_J)
    J1 = J_samples @ e1;  J2 = J_samples @ e2
    cov = np.cov(np.vstack([J1, J2]), bias=False)
    eigvals  = np.linalg.eigvalsh(cov)
    min_var  = np.min(eigvals);  max_var = np.max(eigvals)
    var_Jx   = np.var(J_samples[:, 0], ddof=1)
    var_Jy   = np.var(J_samples[:, 1], ddof=1)
    var_Jz   = np.var(J_samples[:, 2], ddof=1)
    if mean_norm < 1e-12:
        xi2 = np.nan;  xi_db = np.nan
    else:
        xi2   = N * min_var / (mean_norm ** 2)
        xi_db = 10.0 * np.log10(xi2)
    return dict(xi2=xi2, xi_db=xi_db,
                mean_Jx=mean_J[0], mean_Jy=mean_J[1], mean_Jz=mean_J[2],
                mean_J_norm=mean_norm, min_var=min_var, max_var=max_var,
                var_Jx=var_Jx, var_Jy=var_Jy, var_Jz=var_Jz)


@st.cache_data(show_spinner=False)
def simulate_squeezing_cached(n, model, chi, geometry, coupling_range, alpha,
                               tat_jz_over_jy, num_samples, t_max, num_steps, seed):
    N = n * n
    Jx, Jy, Jz = make_hamiltonian_couplings(model, n, chi, geometry,
                                              coupling_range, alpha, tat_jz_over_jy)
    spins = sample_initial_spins(num_samples, N, initial_axis="x", seed=seed)
    times = np.linspace(0.0, t_max, num_steps + 1)
    dt    = times[1] - times[0]
    rows  = []
    for step, t in enumerate(times):
        obs = compute_observables(spins)
        obs.update(time=t, model=model, n=n, N=N)
        rows.append(obs)
        if step < num_steps:
            spins = rk4_step(spins, dt, Jx, Jy, Jz)
    df   = pd.DataFrame(rows)
    best = df.loc[df["xi_db"].idxmin()].to_dict()
    return df, best


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Simulation Inputs")

n = st.sidebar.slider("Lattice size n", min_value=2, max_value=30, value=8, step=1)
N = n * n
st.sidebar.write(f"Total spins: **N = {N}**")

geometry = st.sidebar.selectbox(
    "Coupling geometry",
    ["finite_range", "nearest", "power_law", "all_to_all"],
    index=0,
)
coupling_range = st.sidebar.slider(
    "Coupling range Rc", min_value=1.0, max_value=float(max(2, n)),
    value=min(3.0, float(max(2, n))), step=0.5,
    disabled=(geometry in ["nearest", "all_to_all"]),
)
alpha = st.sidebar.slider(
    "Power-law exponent alpha", min_value=0.5, max_value=6.0, value=3.0, step=0.5,
    disabled=(geometry != "power_law"),
)
chi = st.sidebar.number_input(
    "Twisting strength chi", min_value=0.001, max_value=10.0, value=1.0, step=0.1,
)
tat_jz_over_jy = st.sidebar.slider(
    "TAT magnitude ratio |Jz/Jy|", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
)
t_max = st.sidebar.slider(
    "Maximum simulation time", min_value=0.5, max_value=20.0, value=5.0, step=0.5,
)
num_steps = st.sidebar.slider(
    "Number of time steps", min_value=50, max_value=1000, value=300, step=50,
)
num_samples = st.sidebar.slider(
    "Monte Carlo samples", min_value=100, max_value=10000, value=1000, step=100,
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=999999, value=1234, step=1,
)
run_button = st.sidebar.button("Run simulation", type="primary")


# ============================================================
# Hamiltonian display
# ============================================================

estimated_cost = num_samples * (N ** 2) * num_steps
if estimated_cost > 5e9:
    st.warning(
        "This setting may be slow. Try reducing Monte Carlo samples, time steps, "
        "or lattice size for testing."
    )

st.header("Current model")
st.markdown("OAT:")
st.latex(r"H_{\mathrm{OAT}} = \chi \sum_{i<j} J_{ij} S_i^z S_j^z")
st.markdown("Generalized TAT:")
st.latex(r"H_{\mathrm{TAT}} = \chi \sum_{i<j} J_{ij} \left( r S_i^z S_j^z - S_i^y S_j^y \right)")
st.markdown(r"where $r = |J_z / J_y|$. For ideal symmetric TAT, use $r = 1$.")


# ============================================================
# Run & plot
# ============================================================

if run_button:
    start_time = time.time()

    with st.spinner("Running OAT simulation..."):
        oat_df, oat_best = simulate_squeezing_cached(
            n, "OAT", chi, geometry, coupling_range, alpha,
            tat_jz_over_jy, num_samples, t_max, num_steps, seed,
        )
    with st.spinner("Running TAT simulation..."):
        tat_df, tat_best = simulate_squeezing_cached(
            n, "TAT", chi, geometry, coupling_range, alpha,
            tat_jz_over_jy, num_samples, t_max, num_steps, seed + 1,
        )

    elapsed  = time.time() - start_time
    combined = pd.concat([oat_df, tat_df], ignore_index=True)
    st.success(f"Simulation finished in {elapsed:.2f} seconds.")

    # Best squeezing summary
    st.header("Best Squeezing Summary")
    st.dataframe(pd.DataFrame([
        {"Model": "OAT", "Best squeezing dB": oat_best["xi_db"],
         "Best xi^2": oat_best["xi2"], "Optimal time": oat_best["time"],
         "Min transverse variance": oat_best["min_var"],
         "Mean spin norm": oat_best["mean_J_norm"]},
        {"Model": "TAT", "Best squeezing dB": tat_best["xi_db"],
         "Best xi^2": tat_best["xi2"], "Optimal time": tat_best["time"],
         "Min transverse variance": tat_best["min_var"],
         "Mean spin norm": tat_best["mean_J_norm"]},
    ]), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("OAT best squeezing", f"{oat_best['xi_db']:.3f} dB",
                  f"t = {oat_best['time']:.3f}")
    with col2:
        st.metric("TAT best squeezing", f"{tat_best['xi_db']:.3f} dB",
                  f"t = {tat_best['time']:.3f}")

    # Plot 1: squeezing in dB
    st.header("Squeezing Parameter vs Time")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax1.plot(df["time"], df["xi_db"], label=model_name)
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.set_xlabel(r"Time $t$"); ax1.set_ylabel(r"$10\log_{10}(\xi^2)$ [dB]")
    ax1.set_title("OAT vs TAT squeezing"); ax1.legend(); ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Plot 2: xi^2 linear
    st.header("Linear Squeezing Parameter")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax2.plot(df["time"], df["xi2"], label=model_name)
    ax2.axhline(1.0, linestyle="--", linewidth=1)
    ax2.set_xlabel(r"Time $t$"); ax2.set_ylabel(r"$\xi^2$")
    ax2.set_title(r"Linear squeezing parameter $\xi^2$"); ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Plot 3: transverse variances
    st.header("Transverse Variances")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax3.plot(df["time"], df["min_var"], label=f"{model_name}: min variance")
        ax3.plot(df["time"], df["max_var"], linestyle="--",
                 label=f"{model_name}: max variance")
    ax3.set_xlabel(r"Time $t$"); ax3.set_ylabel("Collective spin variance")
    ax3.set_title("Minimum and maximum transverse variances")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    # Plot 4: raw variances
    st.header("Raw Collective Spin Variances")
    selected_model = st.selectbox("Choose model for raw variance plot", ["OAT", "TAT"])
    raw_df = combined[combined["model"] == selected_model]
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(raw_df["time"], raw_df["var_Jx"], label=r"Var($J_x$)")
    ax4.plot(raw_df["time"], raw_df["var_Jy"], label=r"Var($J_y$)")
    ax4.plot(raw_df["time"], raw_df["var_Jz"], label=r"Var($J_z$)")
    ax4.set_xlabel(r"Time $t$"); ax4.set_ylabel("Variance")
    ax4.set_title(f"Raw collective spin variances: {selected_model}")
    ax4.legend(); ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

    # Export
    st.header("Export Data")
    st.download_button(
        label="Download simulation data as CSV",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name=f"spin_squeezing_n{n}_{geometry}.csv",
        mime="text/csv",
    )

else:
    st.info("Set the parameters in the sidebar, then click **Run simulation**.")
