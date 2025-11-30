import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('flowers9.png')

if img is None:
    print("Erro: Imagem não encontrada.")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r, g, b = cv2.split(img_rgb)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Imagem Original RGB')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(r, cmap='gray')
    plt.title('Plano Vermelho (R)')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(g, cmap='gray')
    plt.title('Plano Verde (G)')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(b, cmap='gray')
    plt.title('Plano Azul (B)')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    cores = ('r', 'g', 'b')
    labels = ('Red', 'Green', 'Blue')
    for i, cor in enumerate(cores):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=cor, label=labels[i])
    plt.title('Histograma RGB')
    plt.legend()
    plt.xlim([0, 256])

    plt.subplot(2, 4, 6)
    plt.hist(r.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('Histograma Canal R')
    plt.xlim([0, 256])

    plt.subplot(2, 4, 7)
    plt.hist(g.ravel(), 256, [0, 256], color='green', alpha=0.7)
    plt.title('Histograma Canal G')
    plt.xlim([0, 256])

    plt.subplot(2, 4, 8)
    plt.hist(b.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Histograma Canal B')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()