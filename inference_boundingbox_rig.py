#!/usr/bin/python3
# coding=utf-8
import glob
import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
# import matplotlib.pyplot as plt

# plt.ion()
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net import F3Net

class Test(object):
    def __init__(self):
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        self.cfg = dataset.Config(datapath = '', snapshot = '/home/reorder/Documents/F3Nett/F3Net/weights/out_v2_2/model-175', mode = 'test')
        #self.cfg = dataset.Config(datapath = '', snapshot = '/home/reorder/Documents/F3Nett/F3Net/weights/out_ver2_1/model-80', mode = 'test')
        self.net = F3Net(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def pred(self, img):
        with torch.no_grad():
            shape = (img.shape[0], img.shape[1])

            # Resize
            img = cv2.resize(img, (200,200))
            # Normalize
            img = (img - self.mean) / self.std
            # ToTensor
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            # On CUDA
            img = img.cuda().float()

            # Network Forward
            img = img.unsqueeze(0)
            out2u = self.net(img, shape)
            # Normalize Output
            pred = (torch.sigmoid(out2u[0, 0]) * 255).cpu().numpy()
            frame_out = pred.astype(np.uint8)
            r, th = cv2.threshold(frame_out, 10, 255, cv2.THRESH_BINARY)
            cnt, hier = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            #cv2.imshow("mask",cnt)
            #v2.waitkey(0)
            res = []
            for i in range(0, len(cnt)):
                box = cv2.boundingRect(cnt[i])
                res.append(box)
            return res, pred

if __name__ == '__main__':

    # # following is the code you need to use to pass input and get the mask output
    files = glob.glob(sys.argv[1] + '/*.png') + glob.glob(sys.argv[1] + '/*.jpg') + glob.glob(sys.argv[1] + '/*.jpeg')
    t = Test()
    start=time.time()
    for idx, i in enumerate(files):
        img = cv2.imread(i, -1)
        resF3, frame_outF3 = t.pred(img)
        print("me")
        if not len(resF3) == 0:
           for boxes in resF3:
               box = boxes
               #label = boxes[0]
               #score = boxes[1]
               sx, sy, w, h = box
               ex = sx + w
               ey = sy + h
               res= cv2.rectangle(img, (sx, sy), (ex, ey), (255, 255, 255), 2)
    cv2.imwrite('/home/reorder/Documents/F3Nett/F3Net/res/Result_InvestorDemo/' + str(idx) + '_.png', res)
    end = time.time()
    print("Time", end-start)   
