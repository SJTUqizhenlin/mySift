import cv2 
import numpy as np 


img_target_color = cv2.imread("targetA.jpg", cv2.IMREAD_COLOR)
img_data_color = cv2.imread("./dataset/3.jpg", cv2.IMREAD_COLOR)
img_target = cv2.imread("targetA.jpg", cv2.IMREAD_GRAYSCALE)
img_data = cv2.imread("./dataset/3.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=200)
kp1, des1 = sift.detectAndCompute(img_target, None)
kp2, des2 = sift.detectAndCompute(img_data, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
nice_match = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        nice_match.append([m])

M = max([img_target.shape[0], img_data.shape[0]])
N = img_target.shape[1] + img_data.shape[1]
img_match = np.zeros((M, N))
img_match = np.zeros(())
img_match = cv2.drawMatchesKnn(img_target_color, kp1, img_data_color, kp2, 
    nice_match, img_match, matchColor=[0,0,255], singlePointColor=[255,0,0])

cv2.imshow("CvMatch", img_match)
cv2.waitKey(0)
