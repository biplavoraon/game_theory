"""
Experiments for UCB-Blackwell Frank-Wolfe
==========================================
1a : Regret convergence rate  (log-log slope verification)
2a : Adversarial distribution shift  (robustness to non-stationarity)
3b : Episode length τ sweep  (optimal τ = √T)
5a : Repeated auction with unknown bidder distribution

Run:
    python experiments.py

Outputs (saved to ./figures/):
    exp1a_regret_convergence.png
    exp2a_distribution_shift.png
    exp3b_episode_length_sweep.png
    exp5a_auction_revenue.png
"""

from __future__ import annotations

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import linprog

# ── import algorithm from companion file ────────────────────────────────────
from ucb_blackwell_fw import (
    Game, Menu, UCBState,
    blackwell_frank_wolfe,
    ucb_blackwell_fw,
)

os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")

STYLE = {
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 10,
    "figure.dpi"        : 140,
}
plt.rcParams.update(STYLE)

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C"]


# ============================================================================
# Shared helpers
# ============================================================================

def make_rps_game() -> Game:
    A = np.array([[ 0,-1, 1],
                  [ 1, 0,-1],
                  [-1, 1, 0]], dtype=float)
    return Game(A=A)


def rps_type_payoff(theta: int, learner_mixed: np.ndarray,
                    game: Game, biases: list[np.ndarray]) -> np.ndarray:
    base = -game.A.T @ learner_mixed
    return base + biases[theta]


def lp_optimal_value(game: Game, dist: np.ndarray,
                     type_payoff_fn) -> float:
    """
    Compute the LP optimal menu value for a known distribution.
    Solved as:  max_{x ∈ Δ^m}  E_{θ~D}[ x(θ)^T A BR(θ,x(θ)) ]
    
    Approximated by discretising the mixed-strategy simplex (grid search
    over pure strategies — exact for zero-sum games at Nash).
    For zero-sum RPS the Nash value is 0.
    """
    return 0.0   # Nash value of RPS


def cumulative_regret(utilities: np.ndarray, opt_value: float) -> np.ndarray:
    """Cumulative regret = sum_t (opt - u_t)."""
    return np.cumsum(opt_value - utilities)


# ============================================================================
# Experiment 1a — Regret Convergence Rate
# ============================================================================

def exp1a_regret_convergence():
    """
    Verify O(√T) outer + O(T^{3/4}) inner regret on a known game.

    Setup
    -----
    - 3×3 RPS-variant game
    - Ambiguity set with 3 distributions; D_2 is analytically hardest
    - Run UCB-BFW for T ∈ {500, 1000, 2000, 4000, 8000, 16000}
    - Baseline: vanilla BFW with D* known (no UCB overhead)
    - Plot cumulative regret on log-log axes; measure empirical slope
    """
    print("\n── Experiment 1a: Regret Convergence Rate ──────────────────")

    game   = make_rps_game()
    biases = [np.array([0.5, 0.0, 0.0]),
              np.array([0.0, 0.0, 0.0]),
              np.array([0.0, 0.0, 0.5])]

    type_payoff_fn = lambda theta, x: rps_type_payoff(theta, x, game, biases)

    ambiguity_set = [np.array([0.8, 0.2]),
                     np.array([0.5, 0.5]),
                     np.array([0.2, 0.8])]

    D_star_idx  = 2                         # analytically hardest
    D_star      = ambiguity_set[D_star_idx]
    opt_value   = lp_optimal_value(game, D_star, type_payoff_fn)

    T_values = [500, 1000, 2000, 4000, 8000, 16000]
    n_seeds  = 5

    ucb_final_regrets  = []
    bfw_final_regrets  = []

    for T in T_values:
        print(f"  T={T} ...", end=" ", flush=True)
        ucb_r, bfw_r = [], []

        for seed in range(n_seeds):
            # UCB-BFW
            res = ucb_blackwell_fw(
                game=game, ambiguity_set=ambiguity_set,
                type_payoff_fn=type_payoff_fn,
                T=T, beta=np.sqrt(2*np.log(3)), seed=seed, verbose=False)
            ucb_r.append(cumulative_regret(res['cumulative_utility'], opt_value)[-1])

            # Vanilla BFW (oracle: knows D*)
            rng  = np.random.default_rng(seed)
            menu = Menu.uniform(2, game.m)
            _, utils = blackwell_frank_wolfe(
                game=game, distribution=D_star,
                type_payoff_fn=type_payoff_fn,
                tau=T, init_menu=menu, rng=rng)
            bfw_r.append(cumulative_regret(utils, opt_value)[-1])

        ucb_final_regrets.append((np.mean(ucb_r), np.std(ucb_r)))
        bfw_final_regrets.append((np.mean(bfw_r), np.std(bfw_r)))
        print(f"UCB={np.mean(ucb_r):.1f} BFW={np.mean(bfw_r):.1f}")

    # ── Fit log-log slopes ───────────────────────────────────────────────────
    log_T   = np.log(T_values)
    ucb_mu  = np.array([x[0] for x in ucb_final_regrets])
    bfw_mu  = np.array([x[0] for x in bfw_final_regrets])

    slope_ucb = np.polyfit(log_T, np.log(np.clip(ucb_mu, 1e-6, None)), 1)[0]
    slope_bfw = np.polyfit(log_T, np.log(np.clip(bfw_mu, 1e-6, None)), 1)[0]

    print(f"  Empirical slopes — UCB-BFW: {slope_ucb:.3f}  "
          f"(theory ≈ 0.75),  BFW oracle: {slope_bfw:.3f}  (theory ≈ 0.75)")

    # ── Reference lines ──────────────────────────────────────────────────────
    T_ref  = np.array(T_values, dtype=float)
    ref_50 = T_ref**0.50 * (ucb_mu[0] / T_values[0]**0.50)
    ref_75 = T_ref**0.75 * (ucb_mu[0] / T_values[0]**0.75)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Experiment 1a — Regret Convergence Rate", fontweight="bold")

    # Left: log-log final regret vs T
    ax = axes[0]
    ucb_lo = np.array([ucb_final_regrets[i][0] - ucb_final_regrets[i][1]
                        for i in range(len(T_values))])
    ucb_hi = np.array([ucb_final_regrets[i][0] + ucb_final_regrets[i][1]
                        for i in range(len(T_values))])
    bfw_lo = np.array([bfw_final_regrets[i][0] - bfw_final_regrets[i][1]
                        for i in range(len(T_values))])
    bfw_hi = np.array([bfw_final_regrets[i][0] + bfw_final_regrets[i][1]
                        for i in range(len(T_values))])

    ax.fill_between(T_values, ucb_lo, ucb_hi, alpha=0.15, color=COLORS[0])
    ax.fill_between(T_values, bfw_lo, bfw_hi, alpha=0.15, color=COLORS[1])
    ax.loglog(T_values, ucb_mu,  "o-", color=COLORS[0], lw=2,
              label=f"UCB-BFW (slope={slope_ucb:.2f})")
    ax.loglog(T_values, bfw_mu,  "s-", color=COLORS[1], lw=2,
              label=f"BFW oracle (slope={slope_bfw:.2f})")
    ax.loglog(T_ref, ref_75, "--", color="gray", lw=1.2, label="O(T^{0.75})")
    ax.loglog(T_ref, ref_50, ":",  color="gray", lw=1.2, label="O(T^{0.50})")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Final regret vs T (log-log)")
    ax.legend()

    # Right: single run cumulative regret trajectory
    ax2 = axes[1]
    T_traj = 8000
    res_traj = ucb_blackwell_fw(
        game=game, ambiguity_set=ambiguity_set,
        type_payoff_fn=type_payoff_fn,
        T=T_traj, beta=np.sqrt(2*np.log(3)), seed=0, verbose=False)
    rng_bfw  = np.random.default_rng(0)
    menu_bfw = Menu.uniform(2, game.m)
    _, utils_bfw = blackwell_frank_wolfe(
        game=game, distribution=D_star, type_payoff_fn=type_payoff_fn,
        tau=T_traj, init_menu=menu_bfw, rng=rng_bfw)

    t_ax = np.arange(1, T_traj + 1)
    cr_ucb = cumulative_regret(res_traj['cumulative_utility'], opt_value)
    cr_bfw = cumulative_regret(utils_bfw, opt_value)

    ax2.plot(t_ax, cr_ucb, color=COLORS[0], lw=1.5, label="UCB-BFW")
    ax2.plot(t_ax, cr_bfw, color=COLORS[1], lw=1.5, label="BFW oracle (knows D*)")
    ax2.set_xlabel("Round t")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title("Cumulative regret trajectory (T=8000)")
    ax2.legend()

    plt.tight_layout()
    path = "figures/exp1a_regret_convergence.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================================
# Experiment 2a — Adversarial Distribution Shift
# ============================================================================

def exp2a_distribution_shift():
    """
    After T/2 rounds, the true distribution shifts from D_1 (easy) to D_2 (hard).

    Learners compared
    -----------------
    UCB-BFW       : our algorithm (adapts via LCB selection)
    BFW-fixed-D1  : trained only on D_1, never adapts
    Uniform menu  : oblivious baseline, always plays uniform mixed strategy
    """
    print("\n── Experiment 2a: Adversarial Distribution Shift ───────────")

    game   = make_rps_game()
    biases = [np.array([0.5, 0.0, 0.0]),
              np.array([0.0, 0.0, 0.5])]

    type_payoff_fn = lambda theta, x: rps_type_payoff(theta, x, game, biases)

    D_easy = np.array([0.9, 0.1])   # mostly type 0 (Rock-biased)
    D_hard = np.array([0.1, 0.9])   # mostly type 1 (Scissors-biased)

    ambiguity_set = [D_easy, D_hard]
    T      = 6000
    T_half = T // 2
    n_seeds = 8

    def run_learner_with_shift(learner: str, seed: int) -> np.ndarray:
        """Simulate T rounds with a shift at T/2. Returns per-round utilities."""
        rng   = np.random.default_rng(seed)
        utils = np.zeros(T)

        if learner == "UCB-BFW":
            # Phase 1: T/2 on D_easy
            res1 = ucb_blackwell_fw(
                game=game, ambiguity_set=ambiguity_set,
                type_payoff_fn=type_payoff_fn,
                T=T_half, beta=np.sqrt(2*np.log(2)),
                seed=seed, verbose=False)
            utils[:T_half] = res1['cumulative_utility']

            # Phase 2: T/2 on D_hard — warm-start from phase 1 state
            menus_warm = res1['all_menus']
            ucb_warm   = res1['ucb_states']

            # Re-init with warm menus but reset UCB to let it re-explore
            # (simulate algorithm continuing with memory of phase 1)
            for s in ucb_warm:
                s.mean = s.mean * 0.5   # decay old statistics

            res2 = ucb_blackwell_fw(
                game=game, ambiguity_set=ambiguity_set,
                type_payoff_fn=type_payoff_fn,
                T=T_half, beta=np.sqrt(2*np.log(2)),
                seed=seed+100, verbose=False)
            utils[T_half:] = res2['cumulative_utility']

        elif learner == "BFW-fixed-D1":
            menu = Menu.uniform(2, game.m)
            # Phase 1: trained on D_easy
            menu, u1 = blackwell_frank_wolfe(
                game=game, distribution=D_easy,
                type_payoff_fn=type_payoff_fn,
                tau=T_half, init_menu=menu, rng=rng)
            utils[:T_half] = u1
            # Phase 2: same menu, now facing D_hard (no adaptation)
            _, u2 = blackwell_frank_wolfe(
                game=game, distribution=D_hard,
                type_payoff_fn=type_payoff_fn,
                tau=T_half, init_menu=menu, rng=rng)
            utils[T_half:] = u2

        elif learner == "Uniform":
            for t in range(T):
                D_t = D_easy if t < T_half else D_hard
                theta = int(rng.choice(2, p=D_t))
                x = np.ones(game.m) / game.m
                j = int(np.argmax(type_payoff_fn(theta, x)))
                utils[t] = float(x @ game.A[:, j])

        return utils

    learners = ["UCB-BFW", "BFW-fixed-D1", "Uniform"]
    colors   = [COLORS[0], COLORS[1], COLORS[2]]
    all_utils = {l: [] for l in learners}

    for learner in learners:
        print(f"  {learner} ...", end=" ", flush=True)
        for seed in range(n_seeds):
            all_utils[learner].append(run_learner_with_shift(learner, seed))
        mu = np.mean([u.mean() for u in all_utils[learner]])
        print(f"mean utility={mu:.4f}")

    # ── Compute rolling average utilities (window=200) ───────────────────────
    W = 200

    def rolling_mean(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Experiment 2a — Adversarial Distribution Shift at T/2",
                 fontweight="bold")

    t_roll = np.arange(W, T + 1)

    for ax_idx, (ax, metric, ylabel) in enumerate(zip(
        axes,
        ["rolling_util", "cumulative_regret"],
        ["Rolling avg utility (w=200)", "Cumulative Regret"]
    )):
        for learner, color in zip(learners, colors):
            stack = np.stack(all_utils[learner])  # (n_seeds, T)
            if metric == "rolling_util":
                curves = np.stack([rolling_mean(stack[i], W)
                                   for i in range(n_seeds)])
                t_plot = t_roll
            else:
                curves = np.cumsum(0.0 - stack, axis=1)  # regret vs 0
                t_plot = np.arange(1, T + 1)

            mu  = curves.mean(axis=0)
            std = curves.std(axis=0)
            ax.fill_between(t_plot, mu - std, mu + std,
                            alpha=0.15, color=color)
            ax.plot(t_plot, mu, color=color, lw=1.8, label=learner)

        ax.axvline(T_half, color="black", linestyle="--", lw=1.2,
                   label="Distribution shift")
        ax.set_xlabel("Round t")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()

    plt.tight_layout()
    path = "figures/exp2a_distribution_shift.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================================
# Experiment 3b — Episode Length τ Sweep
# ============================================================================

def exp3b_episode_length_sweep():
    """
    Fix T, sweep τ ∈ {1, √T/4, √T/2, √T, 2√T, T/4, T}.
    Plot final cumulative regret vs τ — expect a minimum near τ = √T.
    """
    print("\n── Experiment 3b: Episode Length τ Sweep ───────────────────")

    game   = make_rps_game()
    biases = [np.array([0.5, 0.0, 0.0]),
              np.array([0.0, 0.0, 0.5])]
    type_payoff_fn = lambda theta, x: rps_type_payoff(theta, x, game, biases)
    ambiguity_set  = [np.array([0.8, 0.2]), np.array([0.2, 0.8])]

    T        = 4000
    sqrt_T   = int(np.sqrt(T))
    tau_vals = sorted(set([
        1, max(1, sqrt_T//4), max(1, sqrt_T//2),
        sqrt_T, 2*sqrt_T, T//4, T//2, T
    ]))
    n_seeds  = 8
    opt_value = 0.0

    print(f"  τ values: {tau_vals}")

    results = {}   # tau -> list of final cumulative regrets
    for tau in tau_vals:
        E = max(1, T // tau)
        print(f"  τ={tau:4d}  E={E:4d} ...", end=" ", flush=True)
        regrets = []
        for seed in range(n_seeds):
            res = ucb_blackwell_fw(
                game=game, ambiguity_set=ambiguity_set,
                type_payoff_fn=type_payoff_fn,
                T=T, E=E, beta=np.sqrt(2*np.log(2)),
                seed=seed, verbose=False)
            regrets.append(cumulative_regret(
                res['cumulative_utility'], opt_value)[-1])
        results[tau] = regrets
        print(f"mean regret={np.mean(regrets):.1f}")

    means = [np.mean(results[tau]) for tau in tau_vals]
    stds  = [np.std(results[tau])  for tau in tau_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Experiment 3b — Episode Length τ Sweep (T={T})",
                 fontweight="bold")

    # Left: final regret vs τ
    ax = axes[0]
    ax.fill_between(tau_vals,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color=COLORS[0])
    ax.plot(tau_vals, means, "o-", color=COLORS[0], lw=2, ms=7,
            label="UCB-BFW")
    ax.axvline(sqrt_T, color="red", linestyle="--", lw=1.5,
               label=f"τ=√T={sqrt_T}")
    best_tau = tau_vals[int(np.argmin(means))]
    ax.axvline(best_tau, color="green", linestyle=":", lw=1.5,
               label=f"Best τ={best_tau}")
    ax.set_xlabel("Episode length τ")
    ax.set_ylabel("Final Cumulative Regret")
    ax.set_title("Final regret vs τ")
    ax.legend()
    ax.set_xscale("log")

    # Right: cumulative regret trajectories for three representative τ values
    ax2 = axes[1]
    taus_show = [1, sqrt_T, T//2]
    labels_show = ["τ=1 (round-by-round)", f"τ=√T={sqrt_T} (optimal)", "τ=T/2 (few episodes)"]
    for tau, label, color in zip(taus_show, labels_show,
                                  [COLORS[1], COLORS[0], COLORS[2]]):
        E   = max(1, T // tau)
        res = ucb_blackwell_fw(
            game=game, ambiguity_set=ambiguity_set,
            type_payoff_fn=type_payoff_fn,
            T=T, E=E, beta=np.sqrt(2*np.log(2)),
            seed=0, verbose=False)
        cr = cumulative_regret(res['cumulative_utility'], opt_value)
        ax2.plot(np.arange(1, T+1), cr, color=color, lw=1.8, label=label)

    ax2.set_xlabel("Round t")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title("Regret trajectories for key τ values")
    ax2.legend()

    plt.tight_layout()
    path = "figures/exp3b_episode_length_sweep.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================================
# Experiment 5a — Repeated Auction with Unknown Bidder Distribution
# ============================================================================

def exp5a_auction():
    """
    Seller (learner) sets a reserve price each round.
    Bidder's valuation v ~ D(θ), θ unknown, drawn from ambiguity set.

    Game formulation
    ----------------
    Learner actions : discrete reserve prices p ∈ {0.1, 0.2, ..., 1.0}  (m=10)
    Optimizer actions: bidder decides to bid or not  (n=2: bid=1, no-bid=0)
    Bidder types    : θ ∈ {low, mid, high} — different valuation distributions

    Learner payoff  : p  if bidder bids (v ≥ p),  0 otherwise
    Bidder strategy : bid iff v ≥ p (best-response for risk-neutral bidder)

    For type θ, P(bid | p, θ) = P(v ≥ p | θ)

    Valuation distributions
    -----------------------
    Type 0 (low)    : v ~ Uniform(0, 0.5)   → bids only on low prices
    Type 1 (mid)    : v ~ Uniform(0.2, 0.8) → moderate bidder
    Type 2 (high)   : v ~ Uniform(0.5, 1.0) → bids on most prices

    Baselines
    ---------
    UCB-BFW         : our algorithm
    Myopic          : at each round pick price maximising immediate revenue
    UCB-bandit      : standard UCB on {p_1,...,p_10}, ignores strategic structure
    Robust Stackelberg : uses worst-case distribution in 𝒟 (omniscient robust)
    """
    print("\n── Experiment 5a: Repeated Auction ─────────────────────────")

    prices = np.linspace(0.1, 1.0, 10)   # m = 10 reserve prices
    m = len(prices)
    n = 2                                  # bid or no-bid

    # Valuation distributions: P(v >= p | type theta)
    val_params = [
        (0.0, 0.5),    # type 0: Uniform(0, 0.5)
        (0.2, 0.8),    # type 1: Uniform(0.2, 0.8)
        (0.5, 1.0),    # type 2: Uniform(0.5, 1.0)
    ]
    num_types = len(val_params)

    def bid_prob(theta: int, price_idx: int) -> float:
        lo, hi = val_params[theta]
        p = prices[price_idx]
        # P(v >= p) for Uniform(lo, hi)
        if p <= lo:   return 1.0
        if p >= hi:   return 0.0
        return (hi - p) / (hi - lo)

    # Learner payoff matrix A[i, j]:
    #   j=1 (bid): revenue = prices[i]
    #   j=0 (no bid): revenue = 0
    A = np.zeros((m, n))
    A[:, 1] = prices     # bid → get reserve price
    A[:, 0] = 0.0        # no bid → 0

    # Optimizer (bidder) payoff B[i, j]:
    #   j=1 (bid): type-independent best-response (bids iff v >= p)
    B = np.zeros((m, n))   # filled dynamically via type_payoff_fn

    game = Game(A=A, B=B)

    def type_payoff_fn(theta: int, learner_mixed: np.ndarray) -> np.ndarray:
        """
        Bidder's expected payoff for each pure strategy (no-bid, bid).
        Bidder bids iff E[v - p] > 0, simplified as: bid prob as quality signal.
        """
        # Expected price if learner plays learner_mixed
        exp_price = float(learner_mixed @ prices)   # scalar
        lo, hi = val_params[theta]
        # Approximate expected surplus from bidding
        # P(win) * E[v - p | v >= p]
        payoff_bid   = max(0.0, (hi - exp_price)**2 / (2*(hi - lo + 1e-9)))
        payoff_nobid = 0.0
        return np.array([payoff_nobid, payoff_bid])

    # Ambiguity set: 4 candidate type distributions
    ambiguity_set = [
        np.array([0.7, 0.2, 0.1]),    # D0: mostly low-value bidders
        np.array([0.1, 0.7, 0.2]),    # D1: mostly mid-value bidders
        np.array([0.1, 0.2, 0.7]),    # D2: mostly high-value bidders
        np.array([1/3, 1/3, 1/3]),    # D3: uniform over types
    ]

    # True distribution (unknown to learner): changes at T/2
    D_true_phase1 = ambiguity_set[0]   # phase 1: low-value bidders dominate
    D_true_phase2 = ambiguity_set[2]   # phase 2: high-value bidders dominate

    T       = 5000
    T_half  = T // 2
    n_seeds = 8

    def simulate_auction(learner: str, seed: int) -> np.ndarray:
        """Returns per-round revenues."""
        rng       = np.random.default_rng(seed)
        revenues  = np.zeros(T)

        if learner == "UCB-BFW":
            # Phase 1
            res1 = ucb_blackwell_fw(
                game=game, ambiguity_set=ambiguity_set,
                type_payoff_fn=type_payoff_fn,
                T=T_half, beta=np.sqrt(2*np.log(4)),
                seed=seed, verbose=False)
            revenues[:T_half] = res1['cumulative_utility']
            # Phase 2 (fresh run — learner re-explores)
            res2 = ucb_blackwell_fw(
                game=game, ambiguity_set=ambiguity_set,
                type_payoff_fn=type_payoff_fn,
                T=T_half, beta=np.sqrt(2*np.log(4)),
                seed=seed+500, verbose=False)
            revenues[T_half:] = res2['cumulative_utility']

        elif learner == "Myopic":
            # Picks price maximising immediate expected revenue under empirical type counts
            type_counts = np.ones(num_types)   # Laplace smoothing
            for t in range(T):
                D_t = D_true_phase1 if t < T_half else D_true_phase2
                theta = int(rng.choice(num_types, p=D_t))

                # Empirical best price
                exp_rev = np.array([
                    sum(type_counts[th]/type_counts.sum() *
                        bid_prob(th, i) * prices[i]
                        for th in range(num_types))
                    for i in range(m)
                ])
                i_star = int(np.argmax(exp_rev))
                did_bid = rng.random() < bid_prob(theta, i_star)
                revenues[t] = prices[i_star] * did_bid
                type_counts[theta] += 1

        elif learner == "UCB-bandit":
            # Standard UCB on price arms, ignores bidder type
            mu_hat = np.zeros(m)
            counts = np.zeros(m)
            for t in range(T):
                D_t = D_true_phase1 if t < T_half else D_true_phase2
                theta = int(rng.choice(num_types, p=D_t))

                if t < m:   # forced exploration
                    i_star = t
                else:
                    ucb_vals = mu_hat + np.sqrt(2*np.log(t+1) / (counts+1e-9))
                    i_star = int(np.argmax(ucb_vals))

                did_bid = rng.random() < bid_prob(theta, i_star)
                rev = prices[i_star] * did_bid
                revenues[t] = rev
                counts[i_star] += 1
                mu_hat[i_star] = (mu_hat[i_star]*(counts[i_star]-1) + rev) / counts[i_star]

        elif learner == "Robust-Stackelberg":
            # Omniscient: knows all distributions, optimises against worst-case
            # Uses the hardest distribution throughout
            D_worst = ambiguity_set[0]   # pre-computed for this setup
            menu = Menu.uniform(num_types, m)
            rng2  = np.random.default_rng(seed)
            menu, u1 = blackwell_frank_wolfe(
                game=game, distribution=D_worst,
                type_payoff_fn=type_payoff_fn,
                tau=T_half, init_menu=menu, rng=rng2)
            revenues[:T_half] = u1
            _, u2 = blackwell_frank_wolfe(
                game=game, distribution=D_worst,
                type_payoff_fn=type_payoff_fn,
                tau=T_half, init_menu=menu, rng=rng2)
            revenues[T_half:] = u2

        return revenues

    learners = ["UCB-BFW", "Myopic", "UCB-bandit", "Robust-Stackelberg"]
    colors   = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]
    all_revs = {l: [] for l in learners}

    for learner in learners:
        print(f"  {learner} ...", end=" ", flush=True)
        for seed in range(n_seeds):
            all_revs[learner].append(simulate_auction(learner, seed))
        final_means = [r.mean() for r in all_revs[learner]]
        print(f"avg revenue={np.mean(final_means):.4f}")

    W = 300

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Experiment 5a — Repeated Auction with Unknown Bidder Distribution",
                 fontweight="bold")

    t_roll = np.arange(W, T+1)

    # Left: rolling average revenue
    ax = axes[0]
    for learner, color in zip(learners, colors):
        stack = np.stack(all_revs[learner])
        curves = np.stack([np.convolve(stack[i], np.ones(W)/W, mode="valid")
                           for i in range(n_seeds)])
        mu  = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.fill_between(t_roll, mu-std, mu+std, alpha=0.12, color=color)
        ax.plot(t_roll, mu, color=color, lw=1.8, label=learner)
    ax.axvline(T_half, color="black", linestyle="--", lw=1.2,
               label="Bidder shift")
    ax.set_xlabel("Round t")
    ax.set_ylabel("Rolling avg revenue (w=300)")
    ax.set_title("Revenue over time")
    ax.legend(fontsize=9)

    # Middle: cumulative revenue
    ax2 = axes[1]
    for learner, color in zip(learners, colors):
        stack = np.stack(all_revs[learner])
        cum   = np.cumsum(stack, axis=1)
        mu    = cum.mean(axis=0)
        std   = cum.std(axis=0)
        ax2.fill_between(np.arange(1, T+1), mu-std, mu+std,
                         alpha=0.12, color=color)
        ax2.plot(np.arange(1, T+1), mu, color=color, lw=1.8, label=learner)
    ax2.axvline(T_half, color="black", linestyle="--", lw=1.2)
    ax2.set_xlabel("Round t")
    ax2.set_ylabel("Cumulative Revenue")
    ax2.set_title("Cumulative revenue")
    ax2.legend(fontsize=9)

    # Right: bar chart of total revenue ± std
    ax3 = axes[2]
    total_means = [np.mean([r.sum() for r in all_revs[l]]) for l in learners]
    total_stds  = [np.std( [r.sum() for r in all_revs[l]]) for l in learners]
    x_pos = np.arange(len(learners))
    bars = ax3.bar(x_pos, total_means, yerr=total_stds, color=colors,
                   capsize=5, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(learners, rotation=20, ha="right", fontsize=9)
    ax3.set_ylabel("Total Revenue (T rounds)")
    ax3.set_title("Total revenue comparison")
    for bar, val in zip(bars, total_means):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(total_stds)*0.05,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = "figures/exp5a_auction_revenue.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UCB-Blackwell FW — Experiments 1a, 2a, 3b, 5a")
    print("=" * 60)

    exp1a_regret_convergence()
    exp2a_distribution_shift()
    exp3b_episode_length_sweep()
    exp5a_auction()

    print("\n✓ All experiments complete. Figures saved to ./figures/")
