import matplotlib.pyplot as plt
X = make_blobs(n_samples=500, n_features=2, centers=1, cluster_std=1.0, center_box=(-2.0, 2.0))[0]
plt.scatter(X[:,0], X[:,1])
plt.title("Normal distribution in 2D")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

random_vectors = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=5)

random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]
# project the data
X_projected = np.dot(X, random_vectors.T)
# plot the 5 histograms

no_bins = 50

plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.hist(X_projected[:, i], bins=no_bins)
    plt.title(f"Histogram of projected data on vector {i}" )
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

histograms = []
bin_edges = []

for i in range(5):
    histogram, bins = np.histogram(X_projected[:, i], range=[-4, 4], bins=no_bins)
    histograms.append(histogram)
    bin_edges.append(bins)

histograms = np.array(histograms)

probabilities_distributions = []

for i in range(5):
    probabilities = []
    for j in range(no_bins):
        probabilities.append(histograms[i, j] / no_bins)

    probabilities_distributions.append(probabilities)

probabilities_distributions = np.array(probabilities_distributions)

means = np.mean(probabilities_distributions, axis=0)

X_test = np.random.uniform(-3, 3, size=(500, 2))

X_projected_test = np.dot(X_test, random_vectors.T)

probabilities_test = []

for i in range(5):
    dimension = X_projected_test[:, i]
    probabilities = probabilities_distributions[i]
    probabilities_test.append(probabilities[np.digitize(dimension, bin_edges[i]) - 1])

probabilities_test = np.array(probabilities_test).mean(axis=0)
    


# plot
plt.scatter(X_test[:,0], X_test[:, 1], c=probabilities_test, cmap='coolwarm', marker='o', edgecolor='k')
plt.show()
plt.scatter(X[:,0], X[:,1], c='black')

no_bins = 200

plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.hist(X_projected[:, i], bins=no_bins)
    plt.title(f"Histogram of projected data on vector {i}" )
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


histograms = []
bin_edges = []

for i in range(5):
    histogram, bins = np.histogram(X_projected[:, i], range=[-4, 4], bins=no_bins)
    histograms.append(histogram)
    bin_edges.append(bins)

histograms = np.array(histograms)

probabilities_distributions = []

for i in range(5):
    probabilities = []
    for j in range(no_bins):
        probabilities.append(histograms[i, j] / no_bins)

    probabilities_distributions.append(probabilities)

probabilities_distributions = np.array(probabilities_distributions)

means = np.mean(probabilities_distributions, axis=0)


X_test = np.random.uniform(-3, 3, size=(500, 2))

X_projected_test = np.dot(X_test, random_vectors.T)

probabilities_test = []

for i in range(5):
    dimension = X_projected_test[:, i]
    probabilities = probabilities_distributions[i]
    probabilities_test.append(probabilities[np.digitize(dimension, bin_edges[i]) - 1])

probabilities_test = np.array(probabilities_test).mean(axis=0)
    


# plot
plt.scatter(X_test[:,0], X_test[:, 1], c=probabilities_test, cmap='coolwarm', marker='o', edgecolor='k')
plt.show()
plt.scatter(X[:,0], X[:,1], c='black')

