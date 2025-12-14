import numpy as np
from utils import cross_entropy

def pso(model, X_val, y_val, particles=30, iterations=40):
    dim = model.n_weights
    Xp = np.random.randn(particles, dim)
    Vp = np.random.randn(particles, dim) * 0.1

    pbest = Xp.copy()
    pbest_loss = np.array([
        cross_entropy(y_val, model.forward(X_val, p))
        for p in pbest
    ])

    gbest = pbest[np.argmin(pbest_loss)]
    gbest_loss = pbest_loss.min()

    history = []

    for _ in range(iterations):
        for i in range(particles):
            Vp[i] = 0.7*Vp[i] + 1.5*np.random.rand(dim)*(pbest[i]-Xp[i]) \
                    + 1.5*np.random.rand(dim)*(gbest-Xp[i])
            Xp[i] += Vp[i]

            loss = cross_entropy(y_val, model.forward(X_val, Xp[i]))
            if loss < pbest_loss[i]:
                pbest[i] = Xp[i]
                pbest_loss[i] = loss

        gbest = pbest[np.argmin(pbest_loss)]
        history.append(pbest_loss.min())

    return gbest, history
