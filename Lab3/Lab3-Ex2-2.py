from mpl_toolkits.mplot3d import Axes3D
X, _ = make_blobs(n_samples=500, n_features=3, centers=[[0, 10, 0], [10, 0, 10]], cluster_std=1.0)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', alpha=0.7)

ax.set_title("2 Clusters in 3D")

plt.tight_layout()
plt.show()
Iforest = iforest.IForest(contamination=0.02)
Loda = loda.LODA(contamination=0.02)
Dif = dif.DIF(contamination=0.02)

Iforest.fit(X)
Loda.fit(X)
Dif.fit(X)

X_test = np.random.uniform(-10, 20, size=(1000, 3))

y_iforest = Iforest.decision_function(X_test)
y_loda = Loda.decision_function(X_test)
y_dif = Dif.decision_function(X_test)

def plot_3d(title, X, y, cmap):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap, marker='o', edgecolor='k', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label="Decision Function")
    plt.show()

plot_3d("IForest", X_test, y_iforest, cmap='coolwarm')
plot_3d("LODA", X_test, y_loda, cmap='coolwarm')
plot_3d("DIF", X_test, y_dif, cmap='coolwarm')

for architecture in [[16], [16, 32, 16], [16, 32, 32, 16]]:
    Dif = dif.DIF(contamination=0.02, hidden_neurons=architecture)
    Dif.fit(X)

    y_dif = Dif.decision_function(X_test)

    plot_3d(f"DIF {architecture} architecture", X_test, y_dif, cmap='coolwarm')
for bin in [50, 100, 1000]:
    Loda = loda.LODA(contamination=0.02, n_bins=bin, n_random_cuts=1000)
    Loda.fit(X)

    y_loda = Loda.decision_function(X_test)

    plot_3d(f"LODA {bin} bins", X_test, y_loda, cmap='coolwarm')