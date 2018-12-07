import cv2 
import numpy as np 

img_target = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
img_data = cv2.imread("./dataset/3.jpg", cv2.IMREAD_GRAYSCALE)

kp1 = []
kp2 = []
des1 = np.ones((60, 128))
des2 = np.ones((60, 128))
for i in range(60):
    kp1.append(cv2.KeyPoint(i*2, i*2, 1.0))
    des1[i,:] = np.ones((1, 128)) * i
    kp2.append(cv2.KeyPoint(i, i, 1.0))
    des2[i,:] = np.ones((1, 128)) * i
bf = cv2.BFMatcher()
des1 = np.array(des1, dtype=np.float32)
des2 = np.array(des2, dtype=np.float32)
matches = bf.knnMatch(des1, des2, k=2)
nice_match = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        nice_match.append([m])
img_match = np.zeros((100, 100))
img_match = cv2.drawMatchesKnn(img_target, kp1, img_data, kp2, 
    nice_match, img_match, matchColor=[0,0,255], singlePointColor=[0,0,0])
cv2.imshow("Match", img_match)
cv2.waitKey(0)
