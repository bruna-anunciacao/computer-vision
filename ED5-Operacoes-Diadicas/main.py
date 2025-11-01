import cv2
import numpy as np

cap = cv2.VideoCapture('traffic_sequence.mpg')

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, prev_gray)

    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    combined = np.hstack((prev_gray, gray, diff_thresh))
    cv2.imshow('Frame anterior | Frame atual | Diferenca', combined)

    prev_gray = gray.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
