"""
UCB-Blackwell Frank-Wolfe Algorithm
=====================================
Solves the robust repeated game problem where the learner faces an optimizer
whose type is drawn from an *unknown* distribution in an ambiguity set D.

Structure
---------
  Outer loop : LCB selection over ambiguity set {D_1, ..., D_N}
               → identifies the worst-case (hardest) distribution
  Inner loop : Blackwell Frank-Wolfe for tau rounds given selected D*
               → constructs/refines a near-optimal k-sparse menu M*

Reference: Extension of "Optimal No-Regret Learning in General Games"
           (arXiv 2412.18297) to the distributionally robust setting.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable
import warnings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Game:
    """
    Two-player general-sum game.

    Attributes
    ----------
    A : (m, n) array  — learner's payoff matrix
    B : (m, n) array  — optimizer's payoff matrix (defaults to zero-sum: -A)
    m : int           — number of learner pure strategies
    n : int           — number of optimizer pure strategies
    """
    A: np.ndarray           # learner payoff  (m x n)
    B: np.ndarray = None    # optimizer payoff (m x n); None → zero-sum
    m: int = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        self.A = np.array(self.A, dtype=float)
        self.m, self.n = self.A.shape
        if self.B is None:
            self.B = -self.A
        else:
            self.B = np.array(self.B, dtype=float)
        assert self.B.shape == (self.m, self.n), "A and B must have the same shape"


@dataclass
class Menu:
    """
    A menu M maps optimizer types θ ∈ {0,…,|Theta|-1} to mixed strategies
    x(θ) ∈ Delta^m (the learner's m-dimensional simplex).

    Internally stored as a (|Theta|, m) array.
    """
    strategies: np.ndarray   # (|Theta|, m)  — one mixed strategy per type

    @classmethod
    def uniform(cls, num_types: int, m: int) -> "Menu":
        return cls(np.full((num_types, m), 1.0 / m))

    @classmethod
    def random(cls, num_types: int, m: int, rng: np.random.Generator) -> "Menu":
        raw = rng.dirichlet(np.ones(m), size=num_types)
        return cls(raw)

    def prescribe(self, theta: int) -> np.ndarray:
        """Return the mixed strategy prescribed to type theta."""
        return self.strategies[theta]

    def copy(self) -> "Menu":
        return Menu(self.strategies.copy())


@dataclass
class UCBState:
    """Tracks UCB/LCB statistics for one candidate distribution D_i."""
    mean: float = 0.0
    std: float = 1.0
    visits: int = 0

    def lcb(self, beta: float) -> float:
        """Lower confidence bound — used for worst-case selection."""
        if self.visits == 0:
            return -np.inf   # force exploration of unvisited distributions
        return self.mean - beta * self.std / np.sqrt(self.visits)

    def update(self, utilities: np.ndarray) -> None:
        tau = len(utilities)
        new_mean = utilities.mean()
        new_std  = utilities.std() if tau > 1 else self.std
        total    = self.visits + tau
        self.mean  = (self.visits * self.mean + tau * new_mean) / total
        self.std   = new_std
        self.visits = total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex.
    O(m log m) algorithm (Duchi et al. 2008).
    """
    m = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho  = np.nonzero(u * np.arange(1, m + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def optimizer_best_response(game: Game, theta: int, learner_mixed: np.ndarray,
                             type_payoff_fn: Callable[[int, np.ndarray], np.ndarray]
                             ) -> int:
    """
    Optimizer of type theta best-responds to the learner's mixed strategy.
    Returns the pure strategy index j* that maximises E[B(a, j) | theta].

    type_payoff_fn(theta, learner_mixed) → length-n array of optimizer payoffs.
    """
    payoffs = type_payoff_fn(theta, learner_mixed)
    return int(np.argmax(payoffs))


def fw_linear_oracle(game: Game, grad: np.ndarray) -> np.ndarray:
    """
    Frank-Wolfe linear minimisation oracle over Delta^m.

    argmin_{x ∈ Delta^m}  <grad, x>
    = e_{j*}   where j* = argmin_j grad[j]

    Returns the optimal vertex of the simplex (one-hot vector).
    """
    j_star = int(np.argmin(grad))
    s = np.zeros(game.m)
    s[j_star] = 1.0
    return s


def project_to_target_set_halfspace(v: np.ndarray, target_value: float) -> np.ndarray:
    """
    Project vector payoff v onto the approachability target set.

    We use the simplest target set: C = {u : u >= target_value * 1}
    (i.e. coordinate-wise lower bound on per-round utility).

    For the general CSP/menu case this would be replaced by projection
    onto the set of achievable CSP assignments.
    """
    return np.maximum(v, target_value)


# ---------------------------------------------------------------------------
# Inner loop: Blackwell Frank-Wolfe
# ---------------------------------------------------------------------------

def blackwell_frank_wolfe(
    game: Game,
    distribution: np.ndarray,       # probability vector over Theta (length |Theta|)
    type_payoff_fn: Callable,        # (theta, learner_mixed) -> optimizer payoffs (length n)
    tau: int,                        # number of rounds
    init_menu: Menu,                 # warm-start menu
    target_value: float = 0.0,       # approachability target level
    restart_threshold: float = 1e-3, # restart if menu change < threshold (stagnation)
    rng: np.random.Generator = None,
) -> tuple[Menu, np.ndarray]:
    """
    Run Blackwell Frank-Wolfe for `tau` rounds given a fixed distribution.

    Returns
    -------
    refined_menu : Menu     — updated menu after tau rounds
    utilities    : (tau,)   — per-round learner utilities
    """
    if rng is None:
        rng = np.random.default_rng()

    num_types = len(distribution)
    menu = init_menu.copy()
    utilities = np.zeros(tau)

    # Running sum of vector payoffs (for approachability direction)
    avg_payoff = np.zeros(game.m)
    prev_menu_strategies = menu.strategies.copy()

    for t in range(1, tau + 1):
        # ── 2a. Draw opponent type from D*, opponent best-responds ──────────
        theta = int(rng.choice(num_types, p=distribution))
        x_theta = menu.prescribe(theta)          # learner's mixed strategy for θ
        j_star = optimizer_best_response(game, theta, x_theta, type_payoff_fn)

        # ── 2b. Observe vector payoff and compute approachability gradient ──
        # Vector payoff = learner's expected payoff vector under x_theta vs j*
        # (each coordinate = payoff if learner played pure strategy i)
        v_t = game.A[:, j_star]                  # (m,) — payoff column
        avg_payoff = ((t - 1) * avg_payoff + v_t) / t

        pi_v = project_to_target_set_halfspace(avg_payoff, target_value)
        g    = avg_payoff - pi_v                 # violation direction (m,)

        # Scalar utility for this round
        utilities[t - 1] = float(x_theta @ game.A[:, j_star])

        # ── 2c. Frank-Wolfe linear minimisation oracle ──────────────────────
        s = fw_linear_oracle(game, g)            # optimal vertex of Delta^m

        # ── 2d. Frank-Wolfe convex combination update ───────────────────────
        eta = 2.0 / (t + 2)                      # standard FW step size
        new_strategy = (1.0 - eta) * x_theta + eta * s

        # Update menu for this type (and blend all types toward new strategy)
        menu.strategies[theta] = new_strategy

        # ── 2e. Stagnation / divergence restart check ───────────────────────
        if t % max(1, tau // 10) == 0:
            delta = np.max(np.abs(menu.strategies - prev_menu_strategies))
            if delta < restart_threshold and t < tau - 1:
                warnings.warn(f"BFW stagnation at t={t}, restarting menu to uniform.")
                menu = Menu.uniform(num_types, game.m)
            prev_menu_strategies = menu.strategies.copy()

    return menu, utilities


# ---------------------------------------------------------------------------
# Outer loop: UCB over ambiguity set
# ---------------------------------------------------------------------------

def ucb_blackwell_fw(
    game: Game,
    ambiguity_set: list[np.ndarray],  # list of N distributions over Theta
    type_payoff_fn: Callable,          # (theta, learner_mixed) -> optimizer payoffs
    T: int = 10_000,                   # total time horizon
    E: int = None,                     # number of episodes (default: sqrt(T))
    k: int = None,                     # support size hint (unused directly here)
    beta: float = 1.0,                 # UCB exploration constant
    target_value: float = 0.0,         # Blackwell target level
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    UCB-Blackwell Frank-Wolfe algorithm.

    Parameters
    ----------
    game            : Game object with payoff matrices A, B
    ambiguity_set   : list of N distributions D_1,...,D_N over optimizer types
    type_payoff_fn  : callable (theta, x) -> optimizer payoff vector (length n)
    T               : total rounds
    E               : episodes; defaults to int(sqrt(T))
    beta            : LCB exploration constant
    target_value    : Blackwell approachability target
    seed            : random seed

    Returns
    -------
    dict with keys:
        'best_menu'        : Menu — menu for the hardest distribution
        'all_menus'        : list of Menu — one per distribution
        'ucb_states'       : list of UCBState
        'episode_history'  : list of dicts with per-episode stats
        'cumulative_utility': (T,) array of per-round utilities
        'selected_dist_history': (E,) array of which D was selected each episode
    """
    rng = np.random.default_rng(seed)
    N   = len(ambiguity_set)

    if E is None:
        E = max(1, int(np.sqrt(T)))
    tau = max(1, T // E)

    if verbose:
        print(f"UCB-Blackwell FW | T={T}, E={E}, tau={tau}, N={N} distributions")
        print(f"Game: {game.m}x{game.n}, target={target_value}, beta={beta}")
        print("-" * 60)

    # ── Initialization ───────────────────────────────────────────────────────
    num_types = len(ambiguity_set[0])
    ucb_states = [UCBState() for _ in range(N)]
    menus      = [Menu.random(num_types, game.m, rng) for _ in range(N)]

    episode_history       = []
    cumulative_utility    = np.zeros(T)
    selected_dist_history = np.zeros(E, dtype=int)
    round_idx             = 0

    # ── Outer loop: episodes ─────────────────────────────────────────────────
    for e in range(E):
        # Step 1: LCB selection — find the hardest distribution
        lcb_values = [ucb_states[i].lcb(beta) for i in range(N)]
        i_star     = int(np.argmin(lcb_values))
        D_star     = ambiguity_set[i_star]
        selected_dist_history[e] = i_star

        if verbose and (e % max(1, E // 10) == 0):
            visits = [s.visits for s in ucb_states]
            means  = [f"{s.mean:.3f}" for s in ucb_states]
            print(f"Episode {e+1:4d}/{E} | D*=D_{i_star} | "
                  f"visits={visits} | means={means}")

        # Step 2: Blackwell Frank-Wolfe for tau rounds
        rounds_this_episode = min(tau, T - round_idx)
        if rounds_this_episode <= 0:
            break

        refined_menu, utilities = blackwell_frank_wolfe(
            game          = game,
            distribution  = D_star,
            type_payoff_fn= type_payoff_fn,
            tau           = rounds_this_episode,
            init_menu     = menus[i_star],
            target_value  = target_value,
            rng           = rng,
        )

        # Save refined menu (warm-start for next visit to D_{i*})
        menus[i_star] = refined_menu

        # Step 3: Update UCB statistics for D_{i*}
        ucb_states[i_star].update(utilities)

        # Record cumulative utilities
        end_idx = round_idx + rounds_this_episode
        cumulative_utility[round_idx:end_idx] = utilities

        episode_history.append({
            'episode'     : e + 1,
            'dist_selected': i_star,
            'mean_utility': float(utilities.mean()),
            'std_utility' : float(utilities.std()),
            'lcb_values'  : lcb_values,
        })

        round_idx = end_idx

    # ── Output ────────────────────────────────────────────────────────────────
    # Best menu = menu for the most-visited (hardest) distribution
    most_visited = int(np.argmax([s.visits for s in ucb_states]))
    best_menu    = menus[most_visited]

    if verbose:
        print("-" * 60)
        print(f"Done. Most visited distribution: D_{most_visited}")
        visits = [s.visits for s in ucb_states]
        means  = [f"{s.mean:.3f}" for s in ucb_states]
        print(f"Final visits : {visits}")
        print(f"Final means  : {means}")
        cum = np.cumsum(cumulative_utility)
        print(f"Total utility: {cum[-1]:.2f}  |  "
              f"Avg per round: {cum[-1]/T:.4f}")

    return {
        'best_menu'            : best_menu,
        'all_menus'            : menus,
        'ucb_states'           : ucb_states,
        'episode_history'      : episode_history,
        'cumulative_utility'   : cumulative_utility,
        'selected_dist_history': selected_dist_history,
    }


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

def demo_zero_sum():
    """
    Simple zero-sum game demo.

    Game: Rock-Paper-Scissors variant (3x3).
    Ambiguity set: 3 candidate distributions over 2 optimizer types.
      Type 0 = plays pure Rock-biased
      Type 1 = plays pure Scissors-biased
    """
    print("=" * 60)
    print("DEMO: UCB-Blackwell FW on a 3x3 Rock-Paper-Scissors game")
    print("=" * 60)

    # RPS payoff matrix (learner perspective): win=+1, loss=-1, draw=0
    A = np.array([
        [ 0, -1,  1],   # Rock    vs Rock, Paper, Scissors
        [ 1,  0, -1],   # Paper   vs Rock, Paper, Scissors
        [-1,  1,  0],   # Scissors vs Rock, Paper, Scissors
    ], dtype=float)

    game = Game(A=A)

    # Two optimizer types:
    #   Type 0: utility = column 0 of B (prefers playing Rock)
    #   Type 1: utility = column 2 of B (prefers playing Scissors)
    def type_payoff_fn(theta: int, learner_mixed: np.ndarray) -> np.ndarray:
        """Return optimizer's expected payoff for each pure strategy j."""
        # Optimizer payoff = -A (zero-sum), adjusted by type bias
        base = -game.A.T @ learner_mixed  # (n,) — base optimizer payoffs
        if theta == 0:
            bias = np.array([0.5, 0.0, 0.0])   # type 0 prefers Rock
        else:
            bias = np.array([0.0, 0.0, 0.5])   # type 1 prefers Scissors
        return base + bias

    # Ambiguity set: 3 candidate distributions over {type 0, type 1}
    ambiguity_set = [
        np.array([0.8, 0.2]),   # D_0: mostly type 0 (Rock-heavy)
        np.array([0.5, 0.5]),   # D_1: balanced
        np.array([0.2, 0.8]),   # D_2: mostly type 1 (Scissors-heavy)
    ]

    results = ucb_blackwell_fw(
        game           = game,
        ambiguity_set  = ambiguity_set,
        type_payoff_fn = type_payoff_fn,
        T              = 5_000,
        beta           = np.sqrt(2 * np.log(3)),
        target_value   = 0.0,
        seed           = 0,
        verbose        = True,
    )

    print("\nBest menu (strategy per opponent type):")
    for theta, s in enumerate(results['best_menu'].strategies):
        label = ["Rock", "Paper", "Scissors"]
        dist  = ", ".join(f"{label[i]}:{s[i]:.3f}" for i in range(3))
        print(f"  Type {theta}: [{dist}]")

    # Cumulative average utility
    cu = np.cumsum(results['cumulative_utility'])
    T_actual = len(cu)
    avg_utils = cu / np.arange(1, T_actual + 1)
    print(f"\nFinal average utility (last 100 rounds): "
          f"{avg_utils[-100:].mean():.4f}")
    print(f"(Nash value of RPS = 0.0 — algorithm should converge near 0)")

    return results


def demo_general_sum():
    """
    General-sum 2x3 game with a richer ambiguity set.
    """
    print("\n" + "=" * 60)
    print("DEMO: UCB-Blackwell FW on a 2x3 general-sum game")
    print("=" * 60)

    A = np.array([
        [3.0, 1.0, 0.0],
        [0.0, 2.0, 4.0],
    ])
    B = np.array([
        [1.0, 3.0, 2.0],
        [2.0, 1.0, 3.0],
    ])
    game = Game(A=A, B=B)

    # Three optimizer types with different payoff weightings
    type_weights = [
        np.array([1.0, 0.0]),   # type 0: maximise column 0 of B
        np.array([0.0, 1.0]),   # type 1: maximise column 2 of B
        np.array([0.5, 0.5]),   # type 2: balanced
    ]

    def type_payoff_fn(theta: int, learner_mixed: np.ndarray) -> np.ndarray:
        base   = game.B.T @ learner_mixed          # (n,) standard optimizer payoff
        w      = type_weights[theta]
        bias   = np.array([w[0]*2, 0.0, w[1]*2])  # type-specific bonus
        return base + bias

    # 4-distribution ambiguity set
    ambiguity_set = [
        np.array([0.7, 0.2, 0.1]),
        np.array([0.1, 0.7, 0.2]),
        np.array([0.2, 0.1, 0.7]),
        np.array([1/3, 1/3, 1/3]),
    ]

    results = ucb_blackwell_fw(
        game           = game,
        ambiguity_set  = ambiguity_set,
        type_payoff_fn = type_payoff_fn,
        T              = 8_000,
        beta           = np.sqrt(2 * np.log(4)),
        target_value   = 1.0,
        seed           = 1,
        verbose        = True,
    )

    print("\nBest menu:")
    for theta, s in enumerate(results['best_menu'].strategies):
        dist = ", ".join(f"a{i}:{s[i]:.3f}" for i in range(game.m))
        print(f"  Type {theta}: [{dist}]")

    return results


if __name__ == "__main__":
    r1 = demo_zero_sum()
    r2 = demo_general_sum()
