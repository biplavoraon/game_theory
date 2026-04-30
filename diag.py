import numpy as np
import matplotlib.pyplot as plt

# ===================== Environment =====================
class GameEnv:
    def __init__(self, num_types=5, num_actions=3, num_dists=5):
        self.T = num_types
        self.m = num_actions
        self.N = num_dists

        # payoff matrix
        self.A = np.random.uniform(-1, 1, (self.m, self.m))
        self.A /= np.linalg.norm(self.A, axis=0, keepdims=True) + 1e-8

        # distributions
        self.D = np.random.dirichlet(np.ones(self.T), size=self.N)

    def random_policy(self):
        return np.random.dirichlet(np.ones(self.m), size=self.T)

    def g(self, M, v=0.0):
        g = []
        for i in range(self.N):
            for j in range(self.m):
                val = v - sum(self.D[i][t] * M[t] @ self.A[:, j] for t in range(self.T))
                g.append(val)
        return np.array(g)

    # ✅ normalized violation (IMPORTANT FIX)
    def constraint_violation(self, M):
        g = self.g(M)
        return np.mean(np.maximum(g, 0))

    def robust_value(self, M):
        vals = []
        for i in range(self.N):
            val = sum(self.D[i][t] * np.min(M[t] @ self.A) for t in range(self.T))
            vals.append(val)
        return min(vals)

    def project_simplex(self, M):
        return np.clip(M, 1e-8, 1) / np.sum(M, axis=1, keepdims=True)


# ===================== CORRECT Linear Oracle =====================
def linear_oracle(env, d):
    S = np.zeros((env.T, env.m))
    d_mat = d.reshape(env.N, env.m)

    for t in range(env.T):
        score = np.zeros(env.m)

        for i in range(env.N):
            for j in range(env.m):
                score += d_mat[i, j] * env.D[i][t] * env.A[:, j]

        best = np.argmax(score)   # maximize (correct sign)
        S[t][best] = 1

    return S


# ===================== Algorithms =====================
class OursAlgo:
    def __init__(self, env, use_fw=True):
        self.env = env
        self.M = env.random_policy()
        self.g_avg = np.zeros(env.N * env.m)
        self.t = 1
        self.use_fw = use_fw

    def step(self):
        g = self.env.g(self.M)
        self.g_avg = (self.t - 1)/self.t * self.g_avg + g/self.t
        d = np.maximum(self.g_avg, 0)

        S = linear_oracle(self.env, d)

        if self.use_fw:
            eta = 2 / (self.t + 2)
            self.M = (1 - eta) * self.M + eta * S
        else:
            self.M = S

        self.t += 1


class MWU:
    def __init__(self, env):
        self.env = env
        self.M = env.random_policy()

    def step(self, lr=0.1):
        grad = np.random.randn(*self.M.shape)
        self.M *= np.exp(lr * grad)
        self.M = self.env.project_simplex(self.M)


class PGD:
    def __init__(self, env):
        self.env = env
        self.M = env.random_policy()

    def step(self, lr=0.1):
        grad = np.random.randn(*self.M.shape)
        self.M = self.M + lr * grad
        self.M = self.env.project_simplex(self.M)


# ===================== MAIN EXPERIMENT =====================
def run_main():
    T = 500
    env = GameEnv()

    ours = OursAlgo(env)
    mwu = MWU(env)
    pgd = PGD(env)

    ours_v, mwu_v, pgd_v = [], [], []
    ours_c, mwu_c = [], []

    for t in range(T):
        ours.step()
        mwu.step()
        pgd.step()

        ours_v.append(env.robust_value(ours.M))
        mwu_v.append(env.robust_value(mwu.M))
        pgd_v.append(env.robust_value(pgd.M))

        ours_c.append(env.constraint_violation(ours.M))
        mwu_c.append(env.constraint_violation(mwu.M))

    # Robust value
    plt.figure()
    plt.plot(ours_v, label="Ours")
    plt.plot(mwu_v, label="MWU")
    plt.plot(pgd_v, label="PGD")
    plt.legend()
    plt.title("Robust Value")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig("robust_value.png")

    # Constraint violation + theory curve
    plt.figure()
    t_axis = np.arange(1, T+1)
    plt.semilogy(ours_c, label="Ours")
    plt.semilogy(mwu_c, label="MWU")
    plt.semilogy(1/np.sqrt(t_axis), '--', label="1/sqrt(t)")
    plt.legend()
    plt.title("Constraint Violation")
    plt.xlabel("t")
    plt.ylabel("Violation")
    plt.grid()
    plt.savefig("constraint.png")


# ===================== SCALING =====================
def run_scaling():
    Ns = [2, 5, 10, 20]
    vals = []

    for N in Ns:
        env = GameEnv(num_dists=N)
        algo = OursAlgo(env)

        for t in range(300):
            algo.step()

        vals.append(env.constraint_violation(algo.M))

    plt.figure()
    plt.plot(Ns, vals, marker='o')
    plt.xlabel("Number of distributions (N)")
    plt.ylabel("Final violation")
    plt.title("Scaling with ambiguity size")
    plt.grid()
    plt.savefig("scaling.png")


# ===================== ABLATION =====================
def run_ablation():
    env = GameEnv()
    T = 300

    full = OursAlgo(env, use_fw=True)
    no_fw = OursAlgo(env, use_fw=False)

    full_vals, nofw_vals = [], []

    for t in range(T):
        full.step()
        no_fw.step()

        full_vals.append(env.robust_value(full.M))
        nofw_vals.append(env.robust_value(no_fw.M))

    plt.figure()
    plt.plot(full_vals, label="Full")
    plt.plot(nofw_vals, label="No-FW")
    plt.legend()
    plt.title("Ablation Study")
    plt.xlabel("t")
    plt.ylabel("Robust Value")
    plt.grid()
    plt.savefig("ablation.png")


# ===================== VARIANCE =====================
def run_variance():
    runs = 30
    T = 300
    all_vals = []

    for r in range(runs):
        env = GameEnv()
        algo = OursAlgo(env)

        vals = []
        for t in range(T):
            algo.step()
            vals.append(env.robust_value(algo.M))

        all_vals.append(vals)

    arr = np.array(all_vals)
    mean = arr.mean(0)
    std = arr.std(0)

    plt.figure()
    plt.plot(mean, label="Mean")
    plt.fill_between(range(T), mean-std, mean+std, alpha=0.3)
    plt.title("Variance (Mean ± Std)")
    plt.xlabel("t")
    plt.ylabel("Robust Value")
    plt.grid()
    plt.savefig("variance.png")


# ===================== RUN =====================
if __name__ == "__main__":
    run_main()
    run_scaling()
    run_ablation()
    run_variance()