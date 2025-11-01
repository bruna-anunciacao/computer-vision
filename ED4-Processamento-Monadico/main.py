import cv2
import numpy as np
import matplotlib.pyplot as plt

img_uint8 = cv2.imread('lena.pgm', cv2.IMREAD_GRAYSCALE)
img_double = img_uint8.astype(np.float64) / 255.0 

def brightness(img, value=50):
    return cv2.convertScaleAbs(img, alpha=1.1, beta=value)

def contrast(img, alpha=1.5):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def negative(img):
    return 255 - img

def posterisation(img, levels=4):
    img_norm = img / 255.0
    img_quantized = np.floor(img_norm * levels) / levels
    return np.uint8(img_quantized * 255)

bright_img = brightness(img_uint8, value=50)
contrast_img = contrast(img_uint8, alpha=2)
negative_img = negative(img_uint8)
poster_img = posterisation(img_uint8, levels=10)

titles = ['Brightness', 'Contrast', 'Negative', 'Posterisation']
images = [bright_img, contrast_img, negative_img, poster_img]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
