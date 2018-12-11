import cv2 
import numpy as np 


def getGrad(img0):
    img = cv2.GaussianBlur(img0, ksize=(7,7), sigmaX=1.03) * 1.0
    cannyKerX = np.array([[-3,-10,-3],
                            [0,0,0],
                            [3,10,3]])
    cannyKerY = np.array([[-3,0,3],
                            [-10,0,10],
                            [-3,0,3]])
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=cannyKerX, anchor=(-1,-1)) / 16
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=cannyKerY, anchor=(-1,-1)) / 16
    grad_norm = np.power(np.add(np.power(grad_x, 2), np.power(grad_y, 2)) ,0.5)
    grad_direct = np.arctan(np.divide(grad_y, np.add(grad_x, 0.000001)))
    M = grad_norm.shape[0]
    N = grad_norm.shape[1]
    for i in range(M):
        for j in range(N):
            if grad_x[i,j] < 0:
                grad_direct[i,j] = grad_direct[i,j] + np.pi 
            if grad_direct[i,j] < 0:
                grad_direct[i,j] = grad_direct[i,j] + (np.pi*2)
    tmpimg = np.array(img0)
    # cv2.imshow("grayImg", tmpimg)
    # cv2.waitKey(0)
    for i in range(0,M,10):
        for j in range(0,N,10):
            endx = i + int(np.cos(grad_direct[i,j])*5)
            endy = j + int(np.sin(grad_direct[i,j])*5)
            tmpimg = cv2.line(tmpimg, (j,i), (endy,endx), 255)
    # cv2.imshow("theGrad", tmpimg)
    # cv2.waitKey(0)
    return grad_norm, grad_direct

def subGraph(img, direct, x, y, main_direct):
    N = img.shape[0]
    M = img.shape[1]
    ret = np.zeros((16, 16))
    ret2 = np.zeros((16, 16))
    for i in range(-8,8):
        for j in range(-8,8):
            r = np.power((np.power(i,2) + np.power(j,2)), 0.5)
            theta = np.arctan(j * 1.0 / (i + 0.000001)) 
            if i < 0:
                theta += np.pi
            if theta < 0:
                theta += (np.pi * 2)
            theta += main_direct
            if theta > (np.pi * 2):
                theta -= (np.pi * 2)
            rx = x * 1.0 + r * np.cos(theta)
            ry = y * 1.0 + r * np.sin(theta)
            rx = min(int(np.rint(rx)), N-1)
            rx = max(rx, 0)
            ry = min(int(np.rint(ry)), M-1)
            ry = max(ry, 0)
            ret[i+8, j+8] = img[rx, ry]
            ret2[i+8, j+8] = direct[rx, ry] - main_direct
            if ret2[i+8, j+8] < 0:
                ret2[i+8, j+8] = ret2[i+8, j+8] + (np.pi * 2)
    return ret, ret2

def getKPandDES(img, MaxCorner):
    M = img.shape[0]
    N = img.shape[1]
    kp = cv2.goodFeaturesToTrack(img, maxCorners=MaxCorner, 
        qualityLevel=0.03, minDistance=10)
    kp = kp.tolist()
    MaxCorner = len(kp)
    des = np.zeros((MaxCorner, 128))
    grad_norm, grad_direct = getGrad(img)
    main_direct_list = [0.0] * MaxCorner
    for ip in range(MaxCorner):
        p = kp[ip]
        y = int(p[0][0])
        x = int(p[0][1])
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
        sub_norm, sub_direct = subGraph(img, grad_direct, x, y, main_direct)
        main_direct_list[ip] = main_direct
        deslist = [0.0] * 128
        for i in range(16):
            for j in range(16):
                ind = (((i // 4) * 4) + (j // 4)) * 8
                ind += int(sub_direct[i,j] * 4 / np.pi)
                deslist[ind] += sub_norm[i,j]
        des[ip,:] = np.reshape(np.array(deslist), (1,128))
    img_tmp = np.array(img)
    for ip in range(MaxCorner):
        begy = int(kp[ip][0][0])
        begx = int(kp[ip][0][1])
        endy = begy + int(30 * np.sin(main_direct_list[ip]))
        endx = begx + int(30 * np.cos(main_direct_list[ip]))
        img_tmp = cv2.line(img_tmp, (begy, begx), (endy, endx), 255, 2)
    # cv2.imshow("Direction", img_tmp)
    # cv2.waitKey(0)
    kplist = []
    for i in range(MaxCorner):
        x = kp[i][0][0]
        y = kp[i][0][1]
        kplist.append(cv2.KeyPoint(x, y, _size=grad_norm[int(y),int(x)]))
    return kplist, np.array(des, dtype=np.float32)

def getIMGpyramid(img):
    res = [img]
    M = img.shape[0]
    N = img.shape[1]
    minMN = min(M, N)
    maxMN = max(M, N)
    while minMN >= 96:
        nxt = cv2.resize(res[-1], (0,0), fx=0.8, fy=0.8)
        res.append(nxt)
        M = res[-1].shape[0]
        N = res[-1].shape[1]
        minMN = min(M, N)
    while maxMN <= 960:
        nxt = cv2.resize(res[0], (0,0), fx=1.25, fy=1.25)
        res.insert(0, nxt)
        M = res[0].shape[0]
        N = res[0].shape[1]
        maxMN = max(M, N)
    return res

def drawKPDES(img_target, kp1, des1, img_data, kpdes):
    kp2 = kpdes["kp"]
    des2 = kpdes["des"] 
    shape2 = kpdes["gsize"]
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    nice_match = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            nice_match.append([m])
    M = max([img_target.shape[0], img_data.shape[0]])
    N = img_target.shape[1] + img_data.shape[1]
    img_match = np.zeros((M, N))
    kp2New = []
    ratio = shape2[0] / img_data.shape[0]
    for i in range(len(kp2)):
        x = kp2[i][0]
        y = kp2[i][1]
        s = kp2[i][2] 
        kp2New.append(cv2.KeyPoint(x/ratio, y/ratio, _size=s))
    kp2 = kp2New
    img_match = cv2.drawMatchesKnn(img_target, kp1, img_data, kp2, 
        nice_match, img_match, matchColor=[0,0,255], singlePointColor=[255,0,0])
    cv2.imshow("MyMatch", img_match)
    cv2.waitKey(0)
