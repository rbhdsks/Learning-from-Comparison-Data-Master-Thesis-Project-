import numpy as np
def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))
def centre(w):
    return w - w.mean()
def generate_comparisons(true_w, n_comparisons, rng):
    n_items = len(true_w)
    i = rng.integers(n_items, size=n_comparisons)
    j = (i + rng.integers(1, n_items, size=n_comparisons)) % n_items
    i_wins = rng.random(n_comparisons) < sigmoid(true_w[i] - true_w[j])
    return np.column_stack([np.where(i_wins, i, j), np.where(i_wins, j, i)])
def win_matrix(data, n_items):
    A = np.zeros((n_items, n_items))
    np.add.at(A, (data[:, 0], data[:, 1]), 1.0)
    return A
def mle_exists(A):
    def reachable(adj):
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(adj)
    return reachable(A > 0) and reachable(A.T > 0)
def log_likelihood(w, data):
    return np.mean(np.log(sigmoid(w[data[:, 0]] - w[data[:, 1]])))
def fit_bradley_terry(data, n_items, tol=1e-10, max_iter=1000):
    A = win_matrix(data, n_items)
    wins = A.sum(axis=1)
    n_pairs = A + A.T
    np.fill_diagonal(n_pairs, 0.0)
    if not mle_exists(A):
        raise ValueError("comparison graph is not strongly connected; the MLE is not finite")
    gamma = np.ones(n_items)
    for iteration in range(max_iter):
        denom = (n_pairs / (gamma[:, None] + gamma[None, :] + 1e-300)).sum(axis=1)
        new_gamma = wins / denom
        new_gamma /= np.exp(np.mean(np.log(new_gamma)))
        converged = np.max(np.abs(np.log(new_gamma) - np.log(gamma))) < tol
        gamma = new_gamma
        if converged:
            break
    return centre(np.log(gamma)), iteration + 1
def check_generator(data, true_w, n_items, min_count=30):
    A = win_matrix(data, n_items)
    print(f"{'pair':>8} {'n':>6} {'empirical':>10} {'model':>8} {'diff':>7}")
    worst = 0.0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            n = A[i, j] + A[j, i]
            if n < min_count:
                continue
            empirical = A[i, j] / n
            model = sigmoid(true_w[i] - true_w[j])
            worst = max(worst, abs(empirical - model))
            print(f"{i} vs {j:>3} {int(n):>6} {empirical:>10.3f} {model:>8.3f} {empirical - model:>+7.3f}")
    print(f"\nlargest deviation: {worst:.3f}")
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_items = 5
    true_w = centre(np.array([2.0, 1.0, 0.5, -0.5, -1.5]))
    data = generate_comparisons(true_w, n_comparisons=2000, rng=rng)
    print("=== generator check ===")
    check_generator(data, true_w, n_items)
    print("\n=== fit ===")
    w_hat, n_iter = fit_bradley_terry(data, n_items)
    print(f"converged in {n_iter} iterations\n")
    print(f"{'item':>5} {'true':>8} {'fitted':>8} {'error':>8}")
    for k in range(n_items):
        print(f"{k:>5} {true_w[k]:>8.3f} {w_hat[k]:>8.3f} {w_hat[k] - true_w[k]:>+8.3f}")
    print(f"\nRMSE              : {np.sqrt(np.mean((w_hat - true_w) ** 2)):.4f}")
    print(f"ranking recovered : {np.array_equal(np.argsort(-w_hat), np.argsort(-true_w))}")
    print(f"log-lik at truth  : {log_likelihood(true_w, data):.5f}")
    print(f"log-lik at fit    : {log_likelihood(w_hat, data):.5f}")
    try:
        import choix
        w_choix = centre(np.asarray(choix.ilsr_pairwise(n_items, list(map(tuple, data)))))
        print(f"max |mine - choix|: {np.max(np.abs(w_hat - w_choix)):.2e}")
    except ImportError:
        print("choix not installed")
    np.save("bt_comparisons.npy", data)
    np.save("bt_true_w.npy", true_w)
