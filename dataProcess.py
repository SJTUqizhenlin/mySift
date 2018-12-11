import cv2
import numpy as np 
from mySift1 import *
import os
import pickle as pk

def getValidFileName(s):
    l = []
    for c in s:
        if ((c >= '0' and c <= '9') or 
            (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')):
            l.append(c)
    return ''.join(l)

def siftOnGraph(imgName):
    print("Processing {0} ...".format(imgName))
    img_data_color = cv2.imread(imgName, cv2.IMREAD_COLOR)
    img_data = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    dataImgL = getIMGpyramid(img_data)
    KPDESlist = []
    for j in range(len(dataImgL)):
        img_data = dataImgL[j]
        kp2, des2 = getKPandDES(img_data, 200)
        kp2list = []
        for k in range(len(kp2)):
            kp2list.append((kp2[k].pt[0], kp2[k].pt[1], kp2[k].size))
        KPDESlist.append({"kp":kp2list, "des":des2, "gsize":img_data.shape})
    pkname = os.path.join("./dataset", getValidFileName(imgName))
    pkfile = open(pkname+".pkl", "wb")
    ob = {"img":img_data_color, "filename":imgName, "KPDESlist":KPDESlist}
    pk.dump(ob, pkfile)
    pkfile.close()

if __name__=="__main__":
    rootPath = "./dataset"
    for dirpath, dirnames, filenames in os.walk(rootPath):
        for filename in filenames:
            filename = os.path.join(dirpath, filename)
            siftOnGraph(filename)
