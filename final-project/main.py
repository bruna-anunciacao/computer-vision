import cv2
import numpy as np
import csv

def process_husky_pose(video_path):
    H = np.array([
        [0.9236, 0, 1077.2023],
        [0, -0.9102, 1952.5904],
        [0, 0, 1]
    ])
    H_inv = np.linalg.inv(H)

    traj_world = []
    traj_image = []
    traj_angle = []

    cap = cv2.VideoCapture(video_path)

    lower_red1 = np.array([0, 100, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 60])
    upper_red2 = np.array([180, 255, 255])

    kernel = np.ones((5,5), np.uint8)

    min_area = 5000
    max_area = 200000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.add(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_rect = None

        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < min_area or area > max_area:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w_rect, h_rect), angle = rect

                if w_rect == 0 or h_rect == 0:
                    continue

                aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

                if aspect > 1.3:
                    best_cnt = cnt
                    best_rect = rect
                    break

        if best_cnt is not None:
            box = cv2.boxPoints(best_rect)
            box = box.astype(np.int32)
            cv2.drawContours(frame, [box], 0, (0,255,0), 2)

            cX = int(best_rect[0][0])
            cY = int(best_rect[0][1])
            cv2.circle(frame,(cX,cY),4,(255,0,0),-1)

            w_rect, h_rect = best_rect[1]
            angle = best_rect[2]

            if w_rect < h_rect:
                theta = angle + 90
            else:
                theta = angle

            traj_image.append((cX, cY))
            traj_angle.append(theta)

            img_pt = np.array([cX, cY, 1.0])
            world_pt = H_inv @ img_pt
            world_pt /= world_pt[2]

            Xw, Yw = world_pt[0], world_pt[1]
            traj_world.append((Xw, Yw))

        else:
            traj_image.append((np.nan,np.nan))
            traj_angle.append(np.nan)
            traj_world.append((np.nan,np.nan))

        cv2.imshow("Rastreamento Husky", cv2.resize(frame,(800,600)))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open('video2_trajetoria.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame","u","v","X","Y","theta_deg"])
        for i, ((u,v),(X,Y),theta) in enumerate(zip(traj_image, traj_world, traj_angle)):
            writer.writerow([i,u,v,X,Y,theta])

    print("Trajetória salva em video2_trajetoria.csv")

process_husky_pose("video2.MP4")
