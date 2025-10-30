import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("penguins.png", cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(hist, color='black')
plt.title("Histograma")
plt.xlabel("Nível de cinza")
plt.ylabel("Frequência")
plt.axvline(_, color='red', linestyle='--', label=f'Threshold = {int(_)}')
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(binary, cmap='gray')
plt.title("Imagem binária")
plt.axis('off')

plt.tight_layout()
plt.show()
