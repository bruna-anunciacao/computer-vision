import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('flowers4.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Imagem 'flowers4.png' não encontrada.")

img_double = img.astype(np.float64) / 255.0

kU = np.ones((15, 15), dtype=np.float64)
kU /= kU.sum()

plt.figure(figsize=(4, 4))
plt.imshow(kU, cmap='gray')
plt.title('Kernel Médio (kU)')
plt.colorbar()
plt.show()

imU = cv2.filter2D(img_double, -1, kU)

size = 2 * 8 + 1
sigma = 5
x = np.arange(-8, 9)
y = np.arange(-8, 9)
x, y = np.meshgrid(x, y)
kG = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kG /= np.sum(kG)

plt.figure(figsize=(4, 4))
plt.imshow(kG, cmap='gray')
plt.title('Kernel Gaussiano (kG)')
plt.colorbar()
plt.show()

imG = cv2.filter2D(img_double, -1, kG)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_double, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imU, cmap='gray')
plt.title('Filtro Médio (imU)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imG, cmap='gray')
plt.title('Filtro Gaussiano (imG)')
plt.axis('off')

plt.tight_layout()
plt.show()
