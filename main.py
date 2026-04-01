import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# =============================
# GAME SETUP
# =============================
uL = np.array([[0, 3],
               [7, 2]])

uO_list = [
    np.array([[0, 1],
              [0, 1]]),
    np.array([[1, 0],
              [2, 0]])
]

alpha = [0.5, 0.5]
m, n = uL.shape


# =============================
# UTILITIES
# =============================
def project(phi):
    return phi[0].sum(), phi[:, 0].sum()

def payoff(phi):
    return np.sum(phi * uL)

def project_simplex(v):
    v = v.flatten()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(v)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0).reshape(m, n)


# =============================
# MWU (No-Regret)
# =============================
def run_mwu(T=5000, eta=0.1):
    log_w = np.zeros(m)
    phi = np.zeros((m, n))

    for _ in range(T):
        x = np.exp(log_w - np.max(log_w))
        x /= x.sum()

        y = np.zeros(n)
        y[np.argmax(x @ uO_list[0])] = 1

        phi += np.outer(x, y)
        log_w += eta * (uL @ y)

    return phi / T

# =============================
# NO-SWAP REGRET (CE)
# =============================
def run_swap_regret(T=5000, eta=0.1):
    # transition weights
    W = np.ones((m, m))  # W[a, a']

    phi = np.zeros((m, n))

    for _ in range(T):
        # normalize rows
        P = W / W.sum(axis=1, keepdims=True)

        # stationary distribution (for 2x2, simple)
        x = np.linalg.eig(P.T)[1][:, 0].real
        x = np.maximum(x, 0)
        x /= x.sum()

        # adversary best response
        y = np.zeros(n)
        y[np.argmax(x @ uO_list[0])] = 1

        phi += np.outer(x, y)

        # update swap regrets
        reward = uL @ y
        for a in range(m):
            for a2 in range(m):
                W[a, a2] *= np.exp(eta * (reward[a2] - reward[a]))

    return phi / T

def compute_menu_points(phi0, phi1):
    mnsr = compute_MNSR()

    # include LP points
    all_points = mnsr + [phi0, phi1]

    # project to 2D
    proj = np.array([project(p) for p in all_points])

    return proj

def compute_menu_hull(phi0, phi1):
    pts = compute_menu_points(phi0, phi1)

    hull = ConvexHull(pts)

    return pts, hull

def plot_menu_region(phi0, phi1):
    pts, hull = compute_menu_hull(phi0, phi1)

    plt.figure(figsize=(6,6))

    # fill full menu region
    plt.fill(pts[hull.vertices, 0],
             pts[hull.vertices, 1],
             alpha=0.3,
             color="cyan",
             label="Menu region")

    # boundary points
    plt.plot(pts[hull.vertices, 0],
             pts[hull.vertices, 1],
             'k-')

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("P(A)")
    plt.ylabel("P(R)")
    plt.title("Full Menu Region")

    plt.legend()
    plt.grid(True)

    plt.show()

# =============================
# MNSR REGION
# =============================
def compute_MNSR(num=100):
    pts = []
    for p in np.linspace(0, 1, num):
        x = np.array([p, 1-p])
        vals = x @ uL
        best = np.max(vals)

        for j in np.where(np.isclose(vals, best))[0]:
            y = np.zeros(n)
            y[j] = 1
            pts.append(np.outer(x, y))
    return pts


# =============================
# STACKELBERG VALUE
# =============================
def compute_stackelberg_value(uO):
    best = -np.inf

    for a in range(m):
        c = -uO[a]

        A_eq = [np.ones(n)]
        b_eq = [1]

        A_ub, b_ub = [], []
        for a2 in range(m):
            if a2 != a:
                A_ub.append(uL[a2] - uL[a])
                b_ub.append(0)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0,1)]*n, method="highs")

        if res.success:
            best = max(best, -res.fun)

    return best


# =============================
# FULL LP (Algorithm 1)
# =============================
def solve_full_lp():
    k = len(uO_list)
    num_vars = k * m * n

    # objective
    c = np.concatenate([-alpha[i] * uL.flatten() for i in range(k)])

    # equality: each φ_i sums to 1
    A_eq = []
    b_eq = []
    for i in range(k):
        row = np.zeros(num_vars)
        row[i*m*n:(i+1)*m*n] = 1
        A_eq.append(row)
        b_eq.append(1)

    A_ub, b_ub = [], []

    # incentive constraints
    for i in range(k):
        for j in range(k):
            if i != j:
                row = np.zeros(num_vars)
                row[i*m*n:(i+1)*m*n] = -uO_list[i].flatten()
                row[j*m*n:(j+1)*m*n] = uO_list[i].flatten()
                A_ub.append(row)
                b_ub.append(0)

    # Stackelberg constraints
    v = [compute_stackelberg_value(uO) for uO in uO_list]
    for i in range(k):
        row = np.zeros(num_vars)
        row[i*m*n:(i+1)*m*n] = -uO_list[i].flatten()
        A_ub.append(row)
        b_ub.append(-v[i])

    # no-regret constraints
    for i in range(k):
        for a_star in range(m):
            row = np.zeros(num_vars)
            for a in range(m):
                for b in range(n):
                    idx = i*m*n + a*n + b
                    row[idx] = uL[a_star, b] - uL[a, b]
            A_ub.append(row)
            b_ub.append(0)

    res = linprog(c,
                  A_ub=np.array(A_ub),
                  b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq),
                  b_eq=np.array(b_eq),
                  bounds=[(0,1)]*num_vars,
                  method="highs")

    if not res.success:
        raise ValueError(res.message)

    x = res.x
    return [x[i*m*n:(i+1)*m*n].reshape(m,n) for i in range(k)]


# =============================
# BLACKWELL
# =============================

def blackwell_step(phi, phi_proj):
    d = phi - phi_proj

    best_val = float("inf")
    best_x, best_y = None, None

    for p in np.linspace(0,1,101):
        x = np.array([p, 1-p])

        worst_val = -float("inf")
        worst_y = None

        for j in range(n):
            y = np.zeros(n)
            y[j] = 1
            val = np.sum((np.outer(x,y) - phi_proj) * d)

            if val > worst_val:
                worst_val = val
                worst_y = y

        if worst_val < best_val:
            best_val = worst_val
            best_x, best_y = x, worst_y

    return best_x, best_y


def run_blackwell(phi0, phi1, T=5000, gamma=0.05, lam=0.5):
    phi = np.zeros((m,n))
    history = []

    target = compute_MNSR() + [phi0, phi1]

    for _ in range(T):
        proj_phi = min(target, key=lambda p: np.linalg.norm(phi - p))
        x, y = blackwell_step(phi, proj_phi)

        phi_bw = np.outer(x, y)
        phi += gamma * (phi_bw - phi)
        phi = project_simplex(phi)

        history.append(phi.copy())

    return phi, history


# =============================
# PLOTTING
# =============================

def plot_all(phi_nr, phi_bw, phi_lp, phi0, phi1, history, phi_swap):
    plt.figure(figsize=(6,6))

    # MNSR
    pts_mnsr = np.array([project(p) for p in compute_MNSR()])
    hull_mnsr = ConvexHull(pts_mnsr)
    plt.fill(pts_mnsr[hull_mnsr.vertices,0], pts_mnsr[hull_mnsr.vertices,1],
             alpha=0.2, color="purple", label="MNSR")

    pts_menu, hull_menu = compute_menu_hull(phi0, phi1)

    plt.fill(pts_menu[hull_menu.vertices, 0],
             pts_menu[hull_menu.vertices, 1],
             alpha=0.2,
             color="cyan",
             label="Menu")

    # # boundary
    # plt.plot(menu_proj[hull_menu.vertices, 0],
    #          menu_proj[hull_menu.vertices, 1],
    #          color="blue", linestyle="--", linewidth=2)


    # Blackwell trajectory
    traj = np.array([project(p) for p in history])

    # prepend true start
    start = np.array([[0.0, 0.0]])
    traj_full = np.vstack([start, traj])
    plt.plot(traj_full[:, 0],
             traj_full[:, 1],
             color="blue",
             linewidth=2,
             label="Blackwell trajectory")
    # points
    plt.scatter(*project(phi_bw), color="purple", s=140, label="Final BW")
    plt.scatter(*project(phi_lp), color="purple", marker="X", s=160, label="LP optimum")
    plt.scatter(*project(phi_nr), color="green", s=200, edgecolors="black", label="MWU")
    plt.scatter(*project(phi_swap), color="orange", s=180, edgecolors="black", label="Swap-Regret (CE)")

    plt.scatter(*project(phi0), color="red", s=120, label="φ₀")
    plt.scatter(*project(phi1), color="blue", s=120, label="φ₁")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("P(A)")
    plt.ylabel("P(R)")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True)
    plt.title("Blackwell vs LP Optimal")

    plt.savefig("final_plot_lp.png", dpi=150)
    print("Saved final_plot_lp.png")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    phi_nr = run_mwu()

    phi_list = solve_full_lp()
    phi0, phi1 = phi_list

    phi_lp = max(phi_list, key=payoff)

    phi_bw, history = run_blackwell(phi0, phi1)
    phi_swap = run_swap_regret()

    print("\n=== PAYOFFS ===")
    print("MWU:        ", payoff(phi_nr))
    print("Swap-Regret:", payoff(phi_swap))
    print("Blackwell:  ", payoff(phi_bw))
    print("LP optimal: ", payoff(phi_lp))

    plot_all(phi_nr, phi_bw, phi_lp, phi0, phi1, history, phi_swap)