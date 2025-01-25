X, _ =make_blobs(n_samples=500, n_features=2, centers=[[10, 0], [0, 10]], cluster_std=1.0)

# plot the data

plt.scatter(X[:,0], X[:,1])
plt.title("2 clusters in 2D")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

from pyod.models import iforest, loda, dif


Iforest = iforest.IForest(contamination=0.02)
Loda = loda.LODA(contamination=0.02)
Dif = dif.DIF(contamination=0.02)

Iforest.fit(X)
Loda.fit(X)
Dif.fit(X)
X_test = np.random.uniform(-10, 20, size=(1000, 2))

y_iforest = Iforest.decision_function(X_test)
y_loda = Loda.decision_function(X_test)
y_dif = Dif.decision_function(X_test)

plt.title("IForest")
plt.scatter(X_test[:,0], X_test[:,1], c=y_iforest, cmap='coolwarm', marker='o', edgecolor='k')
plt.show()

plt.title("LODA")
plt.scatter(X_test[:,0], X_test[:,1], c=y_loda, cmap='coolwarm', marker='o', edgecolor='k')
plt.show()

plt.title("DIF")
plt.scatter(X_test[:,0], X_test[:,1], c=y_dif, cmap='coolwarm', marker='o', edgecolor='k')
plt.show()

# draw 3 subplots

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

fig.suptitle("DIF")

for i, architecture in enumerate([[16], [16, 32, 16], [16, 32, 32, 16]]):
    Dif = dif.DIF(contamination=0.02, hidden_neurons=architecture)
    Dif.fit(X)

    y_dif = Dif.decision_function(X_test)

    axs[i].scatter(X_test[:,0], X_test[:,1], c=y_dif, cmap='coolwarm', marker='o', edgecolor='k')
    axs[i].set_xlabel("x1")
    axs[i].set_ylabel("x2")
    axs[i].set_title(f'{architecture} architecture')

plt.tight_layout()
plt.show().

# draw 3 subplots

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

fig.suptitle("LODA")

for i, bins in enumerate([50, 100, 1000]):
    Loda = loda.LODA(contamination=0.02, n_bins=bins, n_random_cuts=1000)
    Loda.fit(X)

    y_loda = Loda.decision_function(X_test)

    axs[i].scatter(X_test[:,0], X_test[:,1], c=y_loda, cmap='coolwarm', marker='o', edgecolor='k')
    axs[i].set_xlabel("x1")
    axs[i].set_ylabel("x2")
    axs[i].set_title(f'{bins} bins')

plt.tight_layout()
plt.show()