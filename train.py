# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 21:19
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : train.py
# @Software: PyCharm

#-- coding: utf-8 --
import numpy as np
import tensorflow as tf
import collections
import os
import cv2
import glob
from loss import smooth_l1, focal
from model import create_model

# LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

LABELS = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
        'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
        'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup', 'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
        'cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]


# 获取当前目录
PROJECT_ROOT = os.path.dirname(__file__)

# 定义样本路径
train_ann_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "ann", "*.xml")
train_img_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "img")

ann_dir = os.path.join(PROJECT_ROOT, "data", "ann", "*.xml")
img_dir = os.path.join(PROJECT_ROOT, "data", "img")

val_ann_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "ann", "*.xml")
val_img_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "img")

test_img_file = os.path.join(PROJECT_ROOT, "voc_test_data", "img","*")
train_test_img_file = os.path.join(PROJECT_ROOT, "voc_train_data", "img","*")
batch_size = 16
#subtract_mean = [123, 117, 104]
#divide_by_stddev = 128
# 获取该路径下的xml
train_ann_fnames = glob.glob(train_ann_dir)
ann_fnames = glob.glob(ann_dir)
val_ann_fnames = glob.glob(val_ann_dir)
test_img_fnames = glob.glob(test_img_file)
log_dir =os.path.join(PROJECT_ROOT, "log")


def lr_schedule(epoch):
    if epoch < 10:
        return 0.0001
    elif epoch < 20:
        return 0.00001
    else:
        return 0.00001

class TrainCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print("train begin...")

    def on_batch_end(self, batch, logs={}):
        print(logs)
        pass
    pass


def train(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=[smooth_l1, focal],
                  metrics=None)

    print(model.summary())
    for var in model.variables:
        print(var.name+'___'+str(var.shape))
        pass

    from generator import BatchGenerator
    train_data_generator = BatchGenerator(train_ann_fnames,train_img_dir,LABELS,batch_size)
    val_data_generator = BatchGenerator(val_ann_fnames,val_img_dir,LABELS,batch_size)

    callbacks = [#TrainCallback(),
                 #tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1),
                 #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                 tf.keras.callbacks.ModelCheckpoint(
                     os.path.join(log_dir, "retinanet_voc_{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                     monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,save_freq='epoch')]

    model.fit_generator(train_data_generator, epochs=100, callbacks=callbacks, validation_data=val_data_generator)
    pass

if __name__ ==  '__main__':
    model = create_model(80)
    #model.load_weights(log_dir+"/resnet50_coco_best_v2.1.0.h5")
    #model.load_weights(log_dir + "/resnet50_coco_best_v2.1.0.h5")

    train(model)

    # with h5py.File(log_dir+"/resnet50_coco_best_v2.1.0.h5", 'r') as f:
    #     w = f['model_weights']
    #     for fkey in w.keys():
    #         print(w[fkey], fkey)
    #         print(w[fkey])

    pass