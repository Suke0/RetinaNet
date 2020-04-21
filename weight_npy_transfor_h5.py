# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 21:27
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : weight_npy_transfor_h5.py
# @Software: PyCharm

import numpy as np

def transfor_weight(model):
    vars = model.variables
    # for var in model.variables:
    #     print(var.name.split('/')[-1]+'__'+str(var.shape))
    #     pass
    weights = np.load('./weight/weight.npy', allow_pickle=True)
    for v in weights:
        print(v.shape)
        pass

    arr = []
    for v in vars:
        vname = v.name.split('/')[-1]
        if 'kernel' in vname:
            print(vname + '__' + str(v.shape))
            arr.append(v)
            pass
        pass

    for v in vars:
        vname = v.name.split('/')[-1]
        if 'kernel' not in vname:
            print(vname + '__' + str(v.shape))
            arr.append(v)
            pass
        pass

    for v, v1 in zip(arr, weights):
        v.assign(v1)
        pass

    model.save_weights("./weight/resnet50retinanet_coco.h5")
    pass