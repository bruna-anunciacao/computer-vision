import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("penguins.png", cv2.IMREAD_GRAYSCALE)

Kv = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

Ku = Kv.T

Iv = cv2.filter2D(img, cv2.CV_64F, Kv)
Iu = cv2.filter2D(img, cv2.CV_64F, Ku)

plt.subplot(1, 3, 1)
plt.imshow(Iu, cmap='gray')
plt.title("Gradiente horizontal")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Iv, cmap='gray')
plt.title("Gradiente vertical")
plt.axis('off')
plt.tight_layout()

I = np.sqrt(Iu**2 + Iv**2)
I = np.clip(I, 0, 255).astype(np.uint8)

plt.subplot(1, 3, 3)
plt.imshow(I, cmap='gray')
plt.title("Imagem das bordas")
plt.axis('off')
plt.show()
