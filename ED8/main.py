import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('yellowtargets.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_float = img_rgb.astype(float)
s = np.sum(img_float, axis=2)
s[s == 0] = 1e-5
r = img_float[:, :, 0] / s
g = img_float[:, :, 1] / s

h, w = r.shape
X = np.stack((r.flatten(), g.flatten()), axis=1)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

dist = np.linalg.norm(centers - np.array([0.5, 0.5]), axis=1)
yellow_idx = np.argmin(dist)

mask = (labels == yellow_idx).reshape(h, w).astype(np.uint8) * 255

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask)

print(centroids[1:])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
subset = np.random.choice(len(X), 10000, replace=False)
plt.scatter(X[subset, 0], X[subset, 1], c=labels[subset], cmap='viridis', s=1, alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.show()

plt.figure()
plt.imshow(img_rgb)
plt.scatter(centroids[1:, 0], centroids[1:, 1], c='red', marker='+', s=100)
plt.show()