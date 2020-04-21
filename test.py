# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 13:16
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : test.py
# @Software: PyCharm
import cv2
import numpy as np
from PIL import Image, ImageDraw

from anchor import two_boxes_iou, anchors_for_shape, bbox_transform_env
from image import random_visual_effect_generator, resize_image, read_image_bgr, preprocess_image
from model import create_model

# LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]
from weight_npy_transfor_h5 import transfor_weight

LABELS = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
        'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
        'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup', 'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
        'cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]


def preprocess_img(img_path):
    image = np.asarray(Image.open(img_path).convert('RGB'))
    image = image[:, :, ::-1].copy()

    # visual_effect_generator = random_visual_effect_generator(
    #     contrast_range=(0.9, 1.1),
    #     brightness_range=(-.1, .1),
    #     hue_range=(-0.05, 0.05),
    #     saturation_range=(0.95, 1.05)
    # )
    #
    # visual_effect = next(visual_effect_generator)
    # # apply visual effect
    # image = visual_effect(image_bgr)

    image = image.astype(np.float32)
    image -= [103.939, 116.779, 123.68]

    # resize image
    image, image_scale = resize_image(image)
    return image, image_scale
    pass

#使用NMS方法，对结果去重
def non_max_suppression(bbox_pred,cls_pred, anchors, confidence_threshold=0.5, iou_threshold=0.4):
    # 过滤掉概率小于0.5的预测值
    idxs = np.argmax(cls_pred, axis=-1)
    indexs = range(cls_pred.shape[0])
    cls_pred = cls_pred[indexs, idxs]
    t_index = np.where(cls_pred > confidence_threshold)
    anchors = anchors[t_index]
    cls_pred = np.expand_dims(cls_pred, -1)
    cls_pred = cls_pred[t_index]

    labels_ = idxs[t_index]
    labels_ = np.unique(labels_)
    idxs = np.expand_dims(idxs, -1)
    labels = idxs[t_index]
    bbox_pred = bbox_pred[t_index]

    bbox_pred = bbox_transform_env(anchors,bbox_pred)

    predictions = np.concatenate([labels, cls_pred, bbox_pred], axis=-1)
    result = []
    # print(f'正例样本数：{len(predictions)}')
    for label in labels_:
        idxs = predictions[:, 0] == label
        label_pred_boxes = predictions[idxs]
        while len(label_pred_boxes) > 0:
            idxs = np.argsort(-label_pred_boxes[:, 1])  # 降序排序
            label_max_box = label_pred_boxes[idxs[0]]
            label_pred_boxes = label_pred_boxes[idxs[1:]]
            result.append(label_max_box)
            box1 = label_max_box[2:6]

            for i, box2 in enumerate(label_pred_boxes[:, 2:6]):
                iou = two_boxes_iou(box1, box2)
                if iou > iou_threshold:
                    label_pred_boxes[i, 0] = -1
                    pass
            label_pred_boxes = label_pred_boxes[label_pred_boxes[:, 0] > -1]
            if len(label_pred_boxes) == 1:
                label_pred_boxes = np.reshape(label_pred_boxes, (1, 6))
                pass
            pass
    return np.array(result)  # (n_boxes, 1+1+4)
    pass

# 将级别结果显示在图片上
def draw_boxes(boxes, img_file, cls_names, img_scale):
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)

    for box in boxes:
        box[2:] = box[2:] / img_scale
        draw.rectangle(list(box[2:]), outline='red')
        draw.text(list(box[2:4]),'{} {:.2f}%'.format(cls_names[int(box[0])], box[1] * 100), fill='red')
        print('{} {:.2f}%'.format(cls_names[int(box[0])], box[1] * 100), list(box[2:]))
    img.save(f"output_img.jpg")
    img.show()
    pass



def predict(img_path,cls_names = LABELS):
    # # load image
    # image = read_image_bgr(img_path)
    #
    # # copy to draw on
    # draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    #
    # # preprocess image for network
    # image = preprocess_image(image)
    # image, img_scale = resize_image(image)

    model = create_model(80)
    model.load_weights("./weight/resnet50retinanet_coco.h5")
    #transfor_weight(model)

    vars = model.variables
    # weights = np.load('./weight/weight.npy', allow_pickle=True)
    # arr = []
    print("---------------------------------------------")
    for v in vars:
        print(v.name + '__' + str(v.shape))
        pass
    #
    # for v in vars:
    #     vname = v.name.split('/')[-1]
    #     if 'kernel' not in vname:
    #         print(vname + '__' + str(v.shape))
    #         arr.append(v)
    #         pass
    #     pass
    # print("-------------------------")
    # print(arr[-1])
    # print("-------------------------")
    # print(weights[-1])
    # print("-------------------------")
    image, img_scale = preprocess_img(img_path)
    anchors = anchors_for_shape(image.shape)

    #测试单张图片
    img = np.expand_dims(image,0)
    bbox_pred, cls_pred = model(img)
    bbox_pred = np.squeeze(bbox_pred)
    cls_pred = np.squeeze(cls_pred)
    bboxes = non_max_suppression(bbox_pred, cls_pred, anchors)
    draw_boxes(bboxes,img_path,cls_names,img_scale)
    pass

if __name__ ==  '__main__':
    predict('./test_img/2004374.jpg')
    pass




