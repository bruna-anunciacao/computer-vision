import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

img = cv2.imread('aviao_ed.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

M = cv2.moments(thresh)

area = M['m00']
cx = M['m10'] / area
cy = M['m01'] / area
centroid = (cx, cy)

mu20 = M['mu20']
mu02 = M['mu02']
mu11 = M['mu11']

inertia_matrix = np.array([[mu20, mu11], [mu11, mu02]])

theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
theta_deg = np.degrees(theta)

delta = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
lambda1 = (mu20 + mu02 + delta) / 2
lambda2 = (mu20 + mu02 - delta) / 2

major_axis = 4 * np.sqrt(lambda1 / area)
minor_axis = 4 * np.sqrt(lambda2 / area)

print(f"Area: {area}")
print(f"Centróide: {centroid}")
print(f"Matriz de inercia:\n{inertia_matrix}")
print(f"Orientação: {theta_deg}")

plt.figure(figsize=(8, 6))
plt.imshow(thresh, cmap='gray')
plt.scatter(cx, cy, color='red', marker='+', s=100)

ellipse = Ellipse((cx, cy), width=major_axis, height=minor_axis, angle=theta_deg,
                  edgecolor='red', facecolor='none', linewidth=2)
plt.gca().add_patch(ellipse)

plt.title('Centróide e elipse')
plt.axis('off')
plt.show()