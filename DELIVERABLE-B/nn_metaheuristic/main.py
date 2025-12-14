import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from model import SimpleMLP
from ga import genetic_algorithm
from pso import pso

X_train, X_val, X_test, y_train, y_val, y_test, yoh_train, yoh_val, yoh_test = load_data()

# Baseline
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
mlp.fit(X_train, y_train)
print("Backprop Accuracy:", mlp.score(X_test, y_test))

# Metaheuristics
model = SimpleMLP(4, 10, 3)

ga_weights, ga_curve = genetic_algorithm(model, X_val, yoh_val)
pso_weights, pso_curve = pso(model, X_val, yoh_val)

# Accuracy
def acc(weights):
    preds = model.forward(X_test, weights).argmax(axis=1)
    return accuracy_score(y_test, preds)

print("GA Accuracy:", acc(ga_weights))
print("PSO Accuracy:", acc(pso_weights))

# Plot
plt.plot(ga_curve, label="GA")
plt.plot(pso_curve, label="PSO")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
