import numpy as np
from utils import cross_entropy

def genetic_algorithm(model, X_val, y_val, pop_size=30, generations=40):
    dim = model.n_weights
    population = np.random.randn(pop_size, dim)

    best_losses = []

    for _ in range(generations):
        losses = []
        for ind in population:
            pred = model.forward(X_val, ind)
            loss = cross_entropy(y_val, pred)
            losses.append(loss)

        losses = np.array(losses)
        best_losses.append(losses.min())

        elite = population[np.argsort(losses)[:5]]

        new_pop = elite.copy()
        while len(new_pop) < pop_size:
            p1, p2 = elite[np.random.choice(len(elite), 2)]
            cut = np.random.randint(1, dim)
            child = np.concatenate([p1[:cut], p2[cut:]])
            if np.random.rand() < 0.1:
                child += np.random.randn(dim) * 0.1
            new_pop = np.vstack([new_pop, child])

        population = new_pop

    best = population[np.argmin(losses)]
    return best, best_losses
