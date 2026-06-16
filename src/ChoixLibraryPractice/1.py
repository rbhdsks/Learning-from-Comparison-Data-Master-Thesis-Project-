import choix
import numpy as np
data = [
    (0, [1, 2, 3]),
    (0, [1, 2]),
    (1, [2, 3]),
    (0, [2, 3]),
    (0, [1, 3]),
    (1, [0, 2, 3]),
    (0, [1, 2, 3]),
]
 
n_items = 4
params = choix.ilsr_top1(n_items, data, alpha=0.1)
 
print("Estimated strengths (MNL / Plackett-Luce):", np.round(params, 3))
 
ranking = np.argsort(params)[::-1]
print("Ranking (strongest first):", ranking)
new_choice_set = [0, 1, 2]
probs = choix.probabilities(new_choice_set, params)
 
print(f"\nPredicted probabilities for choice set {new_choice_set}:")
for item, p in zip(new_choice_set, probs):
    print(f"  P(item {item} chosen) = {p:.3f}")
 






