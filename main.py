import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# =============================
# GAME SETUP
# =============================
uL = np.array([[1, -1],
               [-1, 1]])

uO_list = [
    np.array([[1, -1],
              [-1, 1]]),   # type 1 (coordination)

    np.array([[-1, 1],
              [1, -1]])    # type 2 (anti-coordination)
]

alpha = np.array([0.5, 0.5])

m, n = uL.shape
k = len(uO_list)

np.random.seed(0)

# =============================
# HELPERS
# =============================
def unpack(z):
    phi_list = []
    idx = 0
    for _ in range(k):
        phi = z[idx:idx + m*n].reshape(m, n)
        phi_list.append(phi)
        idx += m*n
    x = z[idx:idx + m]
    return phi_list, x

def pack(phi_list, x):
    return np.concatenate([phi.flatten() for phi in phi_list] + [x])

def payoff(phi):
    return np.sum(phi * uL)

# =============================
# CONSTRAINTS (LP)
# =============================
def build_constraints():
    dim = k*m*n + m

    A_eq, b_eq = [], []
    A_ub, b_ub = [], []

    # φ_i sum to 1
    for i in range(k):
        row = np.zeros(dim)
        row[i*m*n:(i+1)*m*n] = 1
        A_eq.append(row)
        b_eq.append(1)

    # link φ_i → x
    for i in range(k):
        for a in range(m):
            row = np.zeros(dim)
            for b in range(n):
                row[i*m*n + a*n + b] = 1
            row[k*m*n + a] = -1
            A_eq.append(row)
            b_eq.append(0)

    # x sums to 1
    row = np.zeros(dim)
    row[k*m*n:] = 1
    A_eq.append(row)
    b_eq.append(1)

    # best-response constraints
    for i in range(k):
        for j in range(n):
            row = np.zeros(dim)
            row[i*m*n:(i+1)*m*n] = uO_list[i].flatten()
            for a in range(m):
                row[k*m*n + a] -= uO_list[i][a, j]
            A_ub.append(-row)
            b_ub.append(0)

    # no-regret
    for a_star in range(m):
        row = np.zeros(dim)
        for i in range(k):
            for a in range(m):
                for b in range(n):
                    row[i*m*n + a*n + b] += alpha[i] * (uL[a_star,b] - uL[a,b])
        A_ub.append(row)
        b_ub.append(0)

    bounds = [(0,1)]*(k*m*n) + [(0,1)]*m
    return dim, A_eq, b_eq, A_ub, b_ub, bounds

dim, A_eq, b_eq, A_ub, b_ub, bounds = build_constraints()

# =============================
# LP CORE
# =============================
def solve_lp():
    c = np.concatenate([-alpha[i]*uL.flatten() for i in range(k)] + [np.zeros(m)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP failed")
    return res.x

def project(z):
    res = linprog(z, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.x if res.success else z

def oracle(direction):
    res = linprog(direction, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("Oracle failed")
    return res.x

def solve_lp_with_duals():
    c = np.concatenate([
        -alpha[i]*uL.flatten() for i in range(k)
    ] + [np.zeros(m)])

    res = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        raise RuntimeError("LP failed")

    # ---- Dual variables ----
    dual_ineq = -res.ineqlin.marginals   # inequality constraints
    dual_eq = res.eqlin.marginals       # equality constraints

    return res.x, dual_ineq, dual_eq

# =============================
# BLACKWELL
# =============================
def run_blackwell(T=1000, gamma=0.05):
    z = np.ones(dim)/dim
    history = []

    for _ in range(T):
        z_proj = project(z)
        d = z_proj - z
        z = z + gamma * (oracle(d) - z)
        history.append(z.copy())

    return z, history

# =============================
# MWU / SWAP
# =============================
def run_mwu_history(T=3000, eta=0.1):
    log_w = np.zeros(m)
    phi = np.zeros((m,n))
    history = []

    for t in range(T):
        x = np.exp(log_w - np.max(log_w))
        x /= x.sum()

        i = np.random.choice(k, p=alpha)
        y = np.zeros(n)
        y[np.argmax(x @ uO_list[i])] = 1

        phi = (t*phi + np.outer(x,y))/(t+1)
        history.append(phi.copy())

        log_w += eta * (uL @ y)

    return history

def run_swap_history(T=3000, eta=0.1):
    logW = np.zeros((m,m))
    phi = np.zeros((m,n))
    history = []

    for t in range(T):
        W = np.exp(logW - np.max(logW, axis=1, keepdims=True))
        P = W / W.sum(axis=1, keepdims=True)

        x = np.ones(m)/m
        for _ in range(20):
            x = P.T @ x
            x /= x.sum()

        i = np.random.choice(k, p=alpha)
        y = np.zeros(n)
        y[np.argmax(x @ uO_list[i])] = 1

        phi = (t*phi + np.outer(x,y))/(t+1)
        history.append(phi.copy())

        reward = uL @ y
        for a in range(m):
            for a2 in range(m):
                logW[a,a2] += eta*(reward[a2]-reward[a])

    return history

# =============================
# METRICS
# =============================
def regret_violation(z):
    phi_list,_ = unpack(z)
    phi_mix = sum(alpha[i]*phi_list[i] for i in range(k))

    max_reg = -1e9
    for a_star in range(m):
        reg = sum(phi_mix[a,b]*(uL[a_star,b]-uL[a,b])
                  for a in range(m) for b in range(n))
        max_reg = max(max_reg, reg)

    return max_reg

def evaluate_history(history):
    pay, reg, dist = [], [], []
    for z in history:
        phi_list,_ = unpack(z)
        phi_mix = sum(alpha[i]*phi_list[i] for i in range(k))

        pay.append(payoff(phi_mix))
        reg.append(regret_violation(z))
        dist.append(np.linalg.norm(z - project(z)))

    return np.array(pay), np.array(reg), np.array(dist)

def evaluate_mwu_history(mwu_hist):
    pay, reg = [], []

    for phi in mwu_hist:
        pay.append(payoff(phi))

        # embed into z-space (approx)
        z_fake = pack([phi]*k, phi.sum(axis=1))
        reg.append(regret_violation(z_fake))

    return np.array(pay), np.array(reg)

def constraint_values(z):
    phi_list, x = unpack(z)

    values = []

    # ---- No-regret constraints ----
    for a_star in range(m):
        val = 0
        for i in range(k):
            for a in range(m):
                for b in range(n):
                    val += alpha[i] * phi_list[i][a,b] * (uL[a_star,b] - uL[a,b])
        values.append(val)

    # ---- Best-response constraints ----
    for i in range(k):
        phi_i = phi_list[i]

        for j in range(n):
            lhs = np.sum(phi_i * uO_list[i])
            rhs = np.dot(x, uO_list[i][:,j])
            values.append(rhs - lhs)  # ≤ 0 form

    return np.array(values)

def track_constraints(history):
    vals = [constraint_values(z) for z in history]
    return np.array(vals)


# =============================
# PLOTTING
# =============================
def plot_payoff(bw_hist, mwu_hist, swap_hist, optimal):
    bw_pay,_,_ = evaluate_history(bw_hist)
    mwu_pay = [payoff(phi) for phi in mwu_hist]
    swap_pay = [payoff(phi) for phi in swap_hist]

    plt.figure()
    plt.plot(bw_pay, label="Blackwell")
    plt.plot(mwu_pay, label="MWU")
    plt.plot(swap_pay, label="Swap")
    plt.axhline(optimal, linestyle='--', label="Optimal")
    plt.legend(); plt.grid()
    plt.title("Payoff convergence")
    plt.savefig("game1_1.png")

def plot_convergence(bw_hist):
    _,reg,dist = evaluate_history(bw_hist)

    plt.figure()
    plt.plot(reg, label="Regret")
    plt.plot(dist, label="Distance")
    plt.legend(); plt.grid()
    plt.title("Blackwell convergence")
    plt.savefig("game1_2.png")

def plot_pca(history):
    Z = np.array(history)
    Zp = PCA(n_components=2).fit_transform(Z)

    plt.figure()
    plt.plot(Zp[:,0], Zp[:,1])
    plt.title("PCA trajectory")
    plt.grid()
    plt.savefig("game1_3.png")

# def plot_3d_full(history, z_lp):
#     xs, ys, zs = [], [], []
#
#     # ---- Blackwell trajectory ----
#     for z in history:
#         phi_list, x = unpack(z)
#
#         xs.append(phi_list[0][0,0])  # φ1
#         ys.append(phi_list[1][0,0])  # φ2
#         zs.append(x[0])              # x
#
#     # ---- Optimal point ----
#     phi_list_lp, x_lp = unpack(z_lp)
#     opt_x = phi_list_lp[0][0,0]
#     opt_y = phi_list_lp[1][0,0]
#     opt_z = x_lp[0]
#
#     # ---- Plot ----
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # trajectory
#     ax.plot(xs, ys, zs, label="Blackwell", linewidth=2)
#
#     # final point
#     ax.scatter(xs[-1], ys[-1], zs[-1], color='purple', s=80, label="BW final")
#
#     # optimal
#     ax.scatter(opt_x, opt_y, opt_z, color='black', s=100, marker='X', label="LP")
#
#     ax.set_xlabel("φ1[0,0]")
#     ax.set_ylabel("φ2[0,0]")
#     ax.set_zlabel("x[0]")
#
#     ax.set_title("3D: Blackwell trajectory + optimal")
#
#     ax.legend()
#     plt.savefig("3d_full.png")

# def plot_menu_3d(ax, phi_list_lp):
#     xs, ys, zs = [], [], []
#
#     for lam in np.linspace(0,1,50):
#         phi1 = phi_list_lp[0]
#         phi2 = phi_list_lp[1]
#
#         phi_mix = lam*phi1 + (1-lam)*phi2
#
#         x = phi_mix.sum(axis=1)
#
#         xs.append(phi_mix[0,0])
#         ys.append(phi_mix[0,0])  # approximate projection
#         zs.append(x[0])
#
#     ax.plot(xs, ys, zs, color="cyan", linewidth=3, label="Menu")

def plot_benchmark(bw_hist, mwu_hist, swap_hist, optimal):

    # Blackwell
    bw_pay, bw_reg, bw_dist = evaluate_history(bw_hist)
    bw_gap = compute_gap(bw_pay, optimal)

    # MWU
    mwu_pay, mwu_reg = evaluate_mwu_history(mwu_hist)
    mwu_gap = compute_gap(mwu_pay, optimal)

    # Swap
    swap_pay = np.array([payoff(phi) for phi in swap_hist])
    swap_gap = compute_gap(swap_pay, optimal)

    plt.figure(figsize=(12,4))

    # ---- GAP ----
    plt.subplot(1,2,1)
    plt.plot(bw_gap, label="Blackwell")
    plt.plot(mwu_gap, label="MWU")
    plt.plot(swap_gap, label="Swap")
    plt.title("Optimality Gap")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    # ---- REGRET ----
    plt.subplot(1,2,2)
    plt.plot(bw_reg, label="Blackwell")
    plt.plot(mwu_reg, label="MWU")
    plt.title("Regret")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    # ---- DISTANCE ----
    # plt.subplot(1,3,3)
    # plt.plot(bw_dist, label="Blackwell")
    # plt.title("Feasibility (Blackwell only)")
    # plt.yscale("log")
    # plt.legend()
    # plt.grid()

    plt.tight_layout()
    plt.savefig("benchmark.png")

def compute_gap(pay, optimal):
    return optimal - pay

def plot_pca_3d(history):
    Z = np.array(history)

    pca = PCA(n_components=3)
    Zp = pca.fit_transform(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(Zp[:,0], Zp[:,1], Zp[:,2])

    ax.set_title("3D PCA projection of trajectory")
    plt.savefig("pca_3d.png")


from sklearn.decomposition import PCA


def build_menu_z(phi_list_lp):
    """
    Build menu in full z-space (φ1, φ2, x)
    """
    Z_menu = []

    for lam in np.linspace(0, 1, 100):
        phi1 = phi_list_lp[0]
        phi2 = phi_list_lp[1]

        phi_mix = lam * phi1 + (1 - lam) * phi2
        x = phi_mix.sum(axis=1)

        # replicate φ_mix across scenarios
        phi_list = [phi_mix.copy() for _ in range(k)]

        z = pack(phi_list, x)
        Z_menu.append(z)

    return np.array(Z_menu)


# def plot_pca_3d_with_menu(history, z_lp, phi_list_lp):
#     # ---- Build data ----
#     Z_traj = np.array(history)
#     Z_menu = build_menu_z(phi_list_lp)
#     Z_opt = z_lp.reshape(1, -1)
#
#     # ---- Combine for PCA ----
#     Z_all = np.vstack([Z_traj, Z_menu, Z_opt])
#
#     # ---- PCA ----
#     pca = PCA(n_components=3)
#     Z_proj = pca.fit_transform(Z_all)
#
#     # split back
#     n_traj = len(Z_traj)
#     n_menu = len(Z_menu)
#
#     traj_proj = Z_proj[:n_traj]
#     menu_proj = Z_proj[n_traj:n_traj+n_menu]
#     opt_proj = Z_proj[-1]
#
#     # ---- Plot ----
#     fig = plt.figure(figsize=(8,6))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # trajectory
#     ax.plot(traj_proj[:,0], traj_proj[:,1], traj_proj[:,2],
#             label="Blackwell", linewidth=2)
#
#     # menu
#     ax.plot(menu_proj[:,0], menu_proj[:,1], menu_proj[:,2],
#             color="cyan", linewidth=3, label="Menu")
#
#     # optimal
#     ax.scatter(opt_proj[0], opt_proj[1], opt_proj[2],
#                color="black", s=100, marker="X", label="LP")
#
#     # final point
#     ax.scatter(traj_proj[-1,0], traj_proj[-1,1], traj_proj[-1,2],
#                color="purple", s=80, label="BW final")
#
#     ax.set_title("3D PCA: Menu vs Blackwell trajectory")
#
#     ax.legend()
#     plt.savefig("pca_3d_menu.png")

def plot_constraint_violations(history):
        vals = track_constraints(history)

        plt.figure(figsize=(8, 5))

        for i in range(vals.shape[1]):
            plt.plot(vals[:, i], label=f"Constraint {i}")

        plt.axhline(0, linestyle='--', color='black')

        plt.title("Constraint violations (projection onto hyperplanes)")
        plt.xlabel("Iteration")
        plt.ylabel("Violation")
        plt.legend()
        plt.grid()

        plt.savefig("constraints.png")

def plot_duals(dual_ineq):
    plt.figure()
    plt.bar(range(len(dual_ineq)), dual_ineq)
    plt.axhline(0, linestyle='--')
    plt.title("Dual variables (inequality constraints)")
    plt.xlabel("Constraint index")
    plt.ylabel("Dual value")
    plt.grid()
    plt.savefig("dual_ineq.png")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    # LP
    z_lp = solve_lp()
    phi_list_lp,_ = unpack(z_lp)
    phi_mix_lp = sum(alpha[i]*phi_list_lp[i] for i in range(k))
    opt = payoff(phi_mix_lp)

    # Blackwell
    z_bw, bw_hist = run_blackwell()
    phi_bw,_ = unpack(z_bw)
    phi_bw = sum(alpha[i]*phi_bw[i] for i in range(k))

    # MWU / Swap
    mwu_hist = run_mwu_history()
    swap_hist = run_swap_history()

    print("\n=== PAYOFFS ===")
    print("LP:", opt)
    print("Blackwell:", payoff(phi_bw))
    print("MWU:", payoff(mwu_hist[-1]))
    print("Swap:", payoff(swap_hist[-1]))

    z_lp, dual_ineq, dual_eq = solve_lp_with_duals()

    print("Active constraint index:", np.argmax(dual_ineq))

    print("\n=== DUAL VARIABLES ===")

    print("\nInequality constraints (A_ub):")
    for i, val in enumerate(dual_ineq):
        print(f"Constraint {i}: {val:.6f}")

    print("\nEquality constraints (A_eq):")
    for i, val in enumerate(dual_eq):
        print(f"Eq {i}: {val:.6f}")


    # Plots
    plot_payoff(bw_hist, mwu_hist, swap_hist, opt)
    plot_convergence(bw_hist)
    plot_pca(bw_hist)
    plot_benchmark(bw_hist, mwu_hist, swap_hist, opt)
    # plot_pca_3d_with_menu(bw_hist, z_lp, phi_list_lp)
    plot_constraint_violations(bw_hist)
    plot_duals(dual_ineq)
