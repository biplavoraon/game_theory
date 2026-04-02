import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
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

# uL = np.array([[2, 0],
#                [0, 1]])
#
# uO_list = [
#     np.array([[0, 2],
#               [1, 0]]),
#     np.array([[2, 0],
#               [0, 3]])
# ]

alpha = np.array([0.5, 0.5])
m, n = uL.shape
k = len(uO_list)

np.random.seed(0)

# =============================
# UTILITIES
# =============================
def project(phi):
    return phi[0].sum(), phi[:, 0].sum()

def payoff(phi):
    return np.sum(phi * uL)

# =============================
# MNSR REGION
# =============================
def compute_MNSR(num=500):
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

MNSR_POINTS = compute_MNSR(num=50)

# =============================
# MENU (φ-space convex hull)
# =============================
def build_menu_vertices(phi0, phi1):
    return MNSR_POINTS + [phi0, phi1]
# =============================
# PROJECTION (φ-space)
# =============================
def convex_projection(phi, vertices):
    P = np.array([p.flatten() for p in vertices])
    phi_flat = phi.flatten()
    k = len(P)

    def obj(lam):
        return np.sum((lam @ P - phi_flat)**2)

    cons = [{"type": "eq", "fun": lambda lam: np.sum(lam) - 1}]
    bounds = [(0,1)] * k
    lam0 = np.ones(k)/k

    res = minimize(obj, lam0, bounds=bounds, constraints=cons)

    if not res.success:
        return phi.copy()

    return (res.x @ P).reshape(phi.shape)

# =============================
# MWU
# =============================
def run_mwu(T=5000, eta=0.1):
    log_w = np.zeros(m)
    phi = np.zeros((m, n))

    for _ in range(T):
        x = np.exp(log_w - np.max(log_w))
        x /= x.sum()

        i = np.random.choice(k, p=alpha)

        y = np.zeros(n)
        y[np.argmax(x @ uO_list[i])] = 1

        phi += np.outer(x, y)
        log_w += eta * (uL @ y)

    return phi / T

# =============================
# SWAP REGRET
# =============================
def run_swap_regret(T=5000, eta=0.1):
    logW = np.zeros((m, m))
    phi = np.zeros((m, n))

    for _ in range(T):
        W = np.exp(logW - np.max(logW, axis=1, keepdims=True))
        P = W / W.sum(axis=1, keepdims=True)

        x = np.ones(m)/m
        for _ in range(20):
            x = P.T @ x
            x /= x.sum()

        i = np.random.choice(k, p=alpha)

        y = np.zeros(n)
        y[np.argmax(x @ uO_list[i])] = 1

        phi += np.outer(x, y)

        reward = uL @ y

        for a in range(m):
            for a2 in range(m):
                logW[a, a2] += eta*(reward[a2] - reward[a])

    return phi / T
# =============================
# LP (CORRECT — SHARED x)
# =============================
def solve_full_lp():
    num_vars = k*m*n + m   # φ_i + x

    c = np.concatenate([
        -alpha[i] * uL.flatten() for i in range(k)
    ] + [np.zeros(m)])

    A_eq, b_eq = [], []

    # φ_i sum to 1
    for i in range(k):
        row = np.zeros(num_vars)
        row[i*m*n:(i+1)*m*n] = 1
        A_eq.append(row)
        b_eq.append(1)

    # link φ_i to shared x
    for i in range(k):
        for a in range(m):
            row = np.zeros(num_vars)

            for b in range(n):
                idx = i * m * n + a * n + b
                row[idx] = 1

            row[k * m * n + a] = -1

            A_eq.append(row)
            b_eq.append(0)
    # x sums to 1
    row = np.zeros(num_vars)
    row[k*m*n:] = 1
    A_eq.append(row)
    b_eq.append(1)

    A_ub, b_ub = [], []

    # best-response constraints
    for i in range(k):
        for j in range(n):
            row = np.zeros(num_vars)

            # LHS: E_{φ_i}[u_O]
            row[i * m * n:(i + 1) * m * n] = uO_list[i].flatten()

            # RHS: x^T u_O(:,j)
            for a in range(m):
                row[k * m * n + a] -= uO_list[i][a, j]

            # enforce: LHS - RHS >= 0  → multiply by -1 for ≤ form
            A_ub.append(-row)
            b_ub.append(0)
    # no-regret constraints
    # CORRECT: ex-ante no-regret
    for a_star in range(m):
        row = np.zeros(num_vars)

        for i in range(k):
            for a in range(m):
                for b in range(n):
                    idx = i * m * n + a * n + b
                    row[idx] += alpha[i] * (uL[a_star, b] - uL[a, b])

        A_ub.append(row)
        b_ub.append(0)

    bounds = [(0,1)]*(k*m*n) + [(0,1)]*m

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        print(res.message)
        raise ValueError("LP failed")

    x = res.x[:k*m*n]
    return [x[i*m*n:(i+1)*m*n].reshape(m,n) for i in range(k)]

# =============================
# BLACKWELL STEP (CORRECT)
# =============================
def blackwell_step(phi, phi_proj):
    d = phi - phi_proj

    best_val = float("inf")
    best_x, best_y = None, None

    for p in np.linspace(0,1,201):
        x = np.array([p,1-p])

        # Bayesian adversary
        payoffs = [
            sum(alpha[i]*(x @ uO_list[i][:,j]) for i in range(k))
            for j in range(n)
        ]
        j_star = np.argmax(payoffs)

        y = np.zeros(n)
        y[j_star] = 1

        val = np.sum((np.outer(x,y) - phi_proj)*d)

        if val < best_val:
            best_val = val
            best_x = x
            best_y = y

    return best_x, best_y

def run_blackwell(phi0, phi1, T=3000, gamma=0.01):
    phi = np.zeros((m,n))
    history = []

    vertices = build_menu_vertices(phi0, phi1)

    for _ in range(T):
        phi_proj = convex_projection(phi, vertices)
        x, y = blackwell_step(phi, phi_proj)

        phi = (1-gamma)*phi + gamma*np.outer(x,y)
        history.append(phi.copy())

    return phi, history

# =============================
# PLOT
# =============================
def plot_all(phi_nr, phi_bw, phi_mix, phi0, phi1, history, phi_swap):
    plt.figure(figsize=(6,6))

    pts = np.array([project(p) for p in MNSR_POINTS])
    hull = ConvexHull(pts)
    plt.fill(pts[hull.vertices,0], pts[hull.vertices,1],
             alpha=0.2, color="purple", label="MNSR")

    verts = build_menu_vertices(phi0, phi1)
    pts_menu = np.array([project(v) for v in verts])
    hull_menu = ConvexHull(pts_menu)
    plt.fill(pts_menu[hull_menu.vertices,0],
             pts_menu[hull_menu.vertices,1],
             alpha=0.2, color="cyan", label="Menu")

    traj = np.array([project(p) for p in history])
    traj = np.vstack([[0,0], traj])
    plt.plot(traj[:,0], traj[:,1], color="blue", label="Blackwell")

    plt.scatter(*project(phi_bw), color="purple", s=120, label="BW final")
    plt.scatter(*project(phi_mix), color="black", marker="X", s=140, label="LP")
    plt.scatter(*project(phi_nr), color="green", s=120, label="MWU")
    plt.scatter(*project(phi_swap), color="orange", s=120, label="Swap")

    plt.scatter(*project(phi0), color="red", s=100, label="φ₀")
    plt.scatter(*project(phi1), color="blue", s=100, label="φ₁")

    plt.legend()
    plt.grid()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("P(A)")
    plt.ylabel("P(R)")
    plt.title("Menu vs MNSR vs Blackwell")

    plt.savefig("game2.png")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    phi_nr = run_mwu()
    phi_swap = run_swap_regret()

    phi_list = solve_full_lp()
    phi0, phi1 = phi_list

    phi_bw, history = run_blackwell(phi0, phi1, T=1000)
    phi_mix = sum(alpha[i]*phi_list[i] for i in range(k))
    lp_value = sum(alpha[i]*payoff(phi_list[i]) for i in range(k))

    print("\n=== PAYOFFS ===")
    print("MWU:        ", payoff(phi_nr))
    print("Swap-Regret:", payoff(phi_swap))
    print("Blackwell:  ", payoff(phi_bw))
    print("LP optimal: ", lp_value)

    plot_all(phi_nr, phi_bw, phi_mix, phi0, phi1, history, phi_swap)