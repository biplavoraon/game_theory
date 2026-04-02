import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

S = 3
A = 2
terminal = 2

# rewards
R = np.array([
    [5, 2],   # state 0
    [3, 4],   # state 1
    [0, 0]    # terminal
])

# transitions P[s,a,s']
P = np.zeros((S, A, S))

# state 0
P[0,0] = [0,1,0]
P[0,1] = [0,0,1]

# state 1
P[1,0] = [0,0,1]
P[1,1] = [1,0,0]

# terminal
P[2,:,2] = 1

epsilon = 0.05

for s in range(S):
    for a in range(A):
        if s != terminal:
            P[s,a] = (1 - epsilon) * P[s,a]
            P[s,a,terminal] += epsilon


def solve_mdp(d, gamma=0.99, iters=100):
    V = np.zeros(S)

    for _ in range(iters):
        Q = np.zeros((S,A))

        for s in range(S):
            for a in range(A):
                Q[s,a] = d[s,a] + gamma * np.dot(P[s,a], V)

        V = np.max(Q, axis=1)

    policy = np.argmax(Q, axis=1)
    return policy


def policy_to_mu(policy, steps=1000):
    mu = np.zeros((S,A))
    state = 0

    for _ in range(steps):
        a = policy[state]
        mu[state,a] += 1

        state = np.random.choice(S, p=P[state,a])

        if state == terminal:
            break

    return mu / np.sum(mu)


def run_blackwell_ssp(T=200, gamma=0.1):
    mu = np.ones((S,A)) / (S*A)
    history = []

    for _ in range(T):
        mu_proj = project_mu(mu)

        d = mu_proj - mu   # direction

        policy = solve_mdp(d)
        mu_oracle = compute_mu_exact(policy)
        mu = (1 - gamma)*mu + gamma*mu_oracle

        history.append(mu.copy())

    return mu, history


def compute_value(mu):
    return np.sum(mu * R)


def compute_value_function(policy, reward, gamma=0.99):
    P_pi = np.zeros((S,S))
    R_pi = np.zeros(S)

    for s in range(S):
        a = policy[s]
        P_pi[s] = P[s,a]
        R_pi[s] = reward[s,a]

    M = np.eye(S) - gamma * P_pi
    V = np.linalg.solve(M, R_pi)

    return V


def plot_ssp(history):
    vals = [compute_value(mu) for mu in history]

    plt.figure()
    plt.plot(vals)
    plt.title("SSP Blackwell Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig("ssp.png")


def compute_mu_exact(policy, gamma=0.99):
    P_pi = np.zeros((S,S))
    for s in range(S):
        a = policy[s]
        P_pi[s] = P[s,a]

    d0 = np.zeros(S)
    d0[0] = 1

    M = np.eye(S) - gamma * P_pi.T
    d = np.linalg.solve(M, d0)

    mu = np.zeros((S,A))
    for s in range(S):
        if s == terminal:
            continue
        mu[s, policy[s]] = d[s]

    return mu


def bellman_residuals(V, reward):
    residuals = np.zeros((S,A))

    for s in range(S):
        for a in range(A):
            residuals[s,a] = reward[s,a] + 0.99*np.dot(P[s,a], V) - V[s]

    return residuals



def plot_bellman(V, reward):
    res = bellman_residuals(V, reward)

    plt.figure()
    for s in range(S):
        plt.plot(res[s], label=f"state {s}")

    plt.axhline(0, linestyle='--')
    plt.title("Bellman residuals")
    plt.xlabel("Action")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid()
    plt.savefig("bellman.png")


def plot_occupancy(mu_final):
    plt.figure()
    plt.imshow(mu_final)
    plt.colorbar()
    plt.title("Occupancy measure")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.savefig("occupancy.png")



def project_mu(mu):
    dim = S * A
    mu_flat = mu.flatten()

    gamma = 0.99

    def obj(z):
        return np.sum((z - mu_flat)**2)

    cons = []

    d0 = np.zeros(S)
    d0[0] = 1

    # -------- flow constraints (ALL states) --------
    for s in range(S):
        def make_constraint(s):
            def constraint(z):
                z_mat = z.reshape(S, A)

                lhs = np.sum(z_mat[s])

                rhs = d0[s]
                for sp in range(S):
                    for ap in range(A):
                        rhs += gamma * z_mat[sp, ap] * P[sp, ap, s]

                return lhs - rhs
            return constraint

        cons.append({'type': 'eq', 'fun': make_constraint(s)})

    # -------- terminal has no actions --------
    for a in range(A):
        def term_constraint(z, a=a):
            return z.reshape(S, A)[terminal, a]

        cons.append({'type': 'eq', 'fun': term_constraint})

    bounds = [(0, None)] * dim

    res = minimize(obj, mu_flat,
                   bounds=bounds,
                   constraints=cons,
                   method='SLSQP')

    if not res.success:
        return mu

    return res.x.reshape(S, A)


if __name__ == "__main__":
    mu_final, history = run_blackwell_ssp()

    print("Final value:", compute_value(mu_final))

    plot_ssp(history)
    policy = np.argmax(mu_final, axis=1)

    plot_occupancy(mu_final)

    d = project_mu(mu_final) - mu_final
    V = compute_value_function(policy, d)

    print("Policy:", policy)
    print("Value function:", V)

    plot_bellman(V, d)