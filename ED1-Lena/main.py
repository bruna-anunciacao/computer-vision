import cv2
import numpy as np
import time

img = cv2.imread("lena.pgm")

frame = img.copy()

y1, y2 = 220, 260
x1, x2 = 310, 360

eyelid = frame[y1:y2, x1:x2].copy()

num_blinks = 3

for _ in range(num_blinks):
    for shift in range(0, 20, 2):
        frame = img.copy()
        frame[y1+shift:y2+shift, x1:x2] = eyelid
        cv2.imshow("Blink Animation", frame)
        cv2.waitKey(50)

    for shift in range(20, 0, -2):
        frame = img.copy()
        frame[y1+shift:y2+shift, x1:x2] = eyelid
        cv2.imshow("Blink Animation", frame)
        cv2.waitKey(50)

    time.sleep(0.3)

cv2.destroyAllWindows()
