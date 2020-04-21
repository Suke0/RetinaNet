# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 11:46
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : demo.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
if __name__ =="__main__":
    cls_pred = np.array([[0,2,3,4],[1,2,1,1],[1,2,3,1],[1,2,3,1]])
    #gt_boxes[index, :4][index, :])
    #idx = np.argmax(cls_pred,-1)
    # print(idx
    # idxs = range(cls_pred.shape[0])
    # cls_pred = cls_pred[idxs,idx]
    # cls_pred = np.expand_dims(cls_pred, -1)
    # print(cls_pred)
    #idx = np.unique(idx)
    #print(idx)
    idx = cls_pred[:,0]==0
    print(idx)
    a = cls_pred[idx]
    a[0,0]=3
    print(a)
    print(cls_pred[[0]])
    pass