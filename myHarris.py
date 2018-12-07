import cv2 
import numpy as np 

img_target = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
img_data = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)

def getGrad(img0):
    img = cv2.GaussianBlur(img0, ksize=(11,11), sigmaX=1.66) * 1.0
    cannyKerX = np.array([[-3,-10,-3],
                            [0,0,0],
                            [3,10,3]])
    cannyKerY = np.array([[-3,0,3],
                            [-10,0,10],
                            [-3,0,3]])
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=cannyKerX, anchor=(-1,-1)) / 16 
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=cannyKerY, anchor=(-1,-1)) / 16
    grad_norm = np.power(np.add(np.power(grad_x, 2), np.power(grad_y, 2)) ,0.5)
    grad_direct = np.divide(grad_y, np.add(grad_x, 0.000001))
    grad_sign = (2 - np.sign(grad_y)) / 2
    grad_sign = np.floor_divide(grad_sign, 0.9)
    grad_direct = (np.arctan(grad_direct) + 
        np.multiply(grad_sign, np.pi) + (np.pi / 2))
    return grad_norm, grad_direct

def getKPandDES(img, MaxCorner):
    M = img.shape[0]
    N = img.shape[1]
    kp = cv2.goodFeaturesToTrack(img, maxCorners=MaxCorner, 
        qualityLevel=0.03, minDistance=10)
    kp = kp.tolist()
    MaxCorner = len(kp)
    des = np.zeros((MaxCorner, 128))
    grad_norm, grad_direct = getGrad(img)
    for ip in range(MaxCorner):
        p = kp[ip]
        x = int(p[0][0])
        y = int(p[0][1])
        if ((x - 8 < 1) or (x + 8 > M - 1) 
            or (y - 8 < 1) or (y + 8 > N - 1)):
            continue
        sub_norm = np.array(grad_norm[x-8:x+8,y-8:y+8])
        sub_direct = np.array(grad_direct[x-8:x+8,y-8:y+8])
        main_direct_bin = np.zeros((1,36))
        for i in range(16):
            for j in range(16):
                ind = int(sub_direct[i,j] * 18 / np.pi)
                main_direct_bin[0,ind] += sub_norm[i,j]
        main_direct = np.argmax(main_direct_bin) * (np.pi/18) + (np.pi/36)
        sub_direct = np.subtract(sub_direct, main_direct)
        for i in range(16):
            for j in range(16):
                if sub_direct[i][j] < 0:
                    sub_direct[i][j] = sub_direct[i][j] + (np.pi * 2)
        deslist = [0.0] * 128
        for i in range(16):
            for j in range(16):
                ind = (((i // 4) * 4) + (j // 4)) * 8
                ind += int(sub_direct[i,j] * 4 / np.pi)
                deslist[ind] += sub_norm[i,j]
        des[ip,:] = np.reshape(np.array(deslist), (1,128))
    kplist = []
    for i in range(MaxCorner):
        x = kp[i][0][0]
        y = kp[i][0][1]
        kplist.append(cv2.KeyPoint(x, y, 1.1))
    return kplist, np.array(des, dtype=np.float32)

kp1, des1 = getKPandDES(img_target, 200)
kp2, des2 = getKPandDES(img_data, 200)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
nice_match = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        nice_match.append([m])
M = max([img_target.shape[0], img_data.shape[0]])
N = img_target.shape[1] + img_data.shape[1]
img_match = np.zeros((M, N))
img_match = np.zeros(())
img_match = cv2.drawMatchesKnn(img_target, kp1, img_data, kp2, 
    nice_match, img_match, matchColor=[0,0,255], singlePointColor=[255,0,0])
cv2.imshow("MyMatch", img_match)
cv2.waitKey(0)
