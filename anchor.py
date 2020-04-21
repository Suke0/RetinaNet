# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 18:59
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : anchor.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import collections
""" Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """
def create_anchor_targets(image_group,box_group,num_classes,negative_overlap=0.4,positive_overlap=0.5):
    #box_group.shape=[batch_size,n_boxes,4+1]
    batch_size = len(image_group)
    max_shape = tuple(max(image.shape[i] for image in image_group) for i in range(3))
    anchors = anchors_for_shape(max_shape)

    regression_batch = np.zeros((batch_size,anchors.shape[0],4 + 1),dtype=np.float32) #the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg)
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=np.float32) #the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg)

    for index, (image,gt_boxes) in enumerate(zip(image_group,box_group)):
        overlaps = compute_overlap(anchors.astype(np.float32),gt_boxes[:,0:4].astype(np.float32))
        argmax_overlaps_inds = np.argmax(overlaps,axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        positive_indices = max_overlaps >= positive_overlap
        ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

        # obtain indices of gt annotations with the greatest overlap
        labels_batch[index, ignore_indices,-1] = -1 #the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg)
        labels_batch[index, positive_indices ,-1] = 1
        regression_batch[index, ignore_indices, -1] = -1 #the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg)
        regression_batch[index, positive_indices, -1] = 1

        # compute target class labels
        labels_batch[index, positive_indices, gt_boxes[:,-1][argmax_overlaps_inds[positive_indices]].astype(int)] = 1
        #regression_batch[index, :, :-1] = bbox_transform(anchors, gt_boxes[:,0:4][argmax_overlaps_inds, :])
        mean = np.array([0, 0, 0, 0])
        std = np.array([0.2, 0.2, 0.2, 0.2])

        regression_batch[index, :, :-1] = np.stack(((gt_boxes[:,0:4][argmax_overlaps_inds, :][:, 0] - anchors[:, 0]) / (anchors[:, 2] - anchors[:, 0]),
                            (gt_boxes[:,0:4][argmax_overlaps_inds, :][:, 1] - anchors[:, 1]) / (anchors[:, 3] - anchors[:, 1]),
                            (gt_boxes[:,0:4][argmax_overlaps_inds, :][:, 2] - anchors[:, 2]) / (anchors[:, 2] - anchors[:, 0]),
                            (gt_boxes[:,0:4][argmax_overlaps_inds, :][:, 3] - anchors[:, 3]) / (anchors[:, 3] - anchors[:, 1]))).T

        regression_batch[index, :, :-1] = (regression_batch[index, :, :-1] - mean) / std

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])
            del anchors_centers

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1
            pass
        pass

    return regression_batch, labels_batch
    pass

def bbox_transform_env(anchors, bbox_pred, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    # The Mean and std are calculated from COCO dataset.
    # Bounding box normalization was firstly introduced in the Fast R-CNN paper.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825  for more details
    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    bbox_pred = bbox_pred * std + mean

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    pred_x1 = bbox_pred[:, 0] * anchor_widths + anchors[:, 0]
    pred_y1 = bbox_pred[:, 1] * anchor_heights + anchors[:, 1]
    pred_x2 = bbox_pred[:, 2] * anchor_widths + anchors[:, 2]
    pred_y2 = bbox_pred[:, 3] * anchor_heights + anchors[:,3]

    pred = np.stack((pred_x1, pred_y1, pred_x2, pred_y2))
    pred = pred.T
    return pred
    pass

def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    # The Mean and std are calculated from COCO dataset.
    # Bounding box normalization was firstly introduced in the Fast R-CNN paper.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825  for more details
    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    # anchor_widths  = anchors[:, 2] - anchors[:, 0]
    # anchor_heights = anchors[:, 3] - anchors[:, 1]

    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details

    # targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / (anchors[:, 2] - anchors[:, 0])
    # targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / (anchors[:, 3] - anchors[:, 1])
    # targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / (anchors[:, 2] - anchors[:, 0])
    # targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / (anchors[:, 3] - anchors[:, 1])

    targets = np.stack(((gt_boxes[:, 0] - anchors[:, 0]) / (anchors[:, 2] - anchors[:, 0]),
                        (gt_boxes[:, 1] - anchors[:, 1]) / (anchors[:, 3] - anchors[:, 1]),
                        (gt_boxes[:, 2] - anchors[:, 2]) / (anchors[:, 2] - anchors[:, 0]),
                        (gt_boxes[:, 3] - anchors[:, 3]) / (anchors[:, 3] - anchors[:, 1])))
    targets = targets.T
    targets = (targets - mean) / std

    return targets
    pass


def compute_overlap(boxes,gt_boxes):
    ious = np.zeros((len(boxes),len(gt_boxes)))
    for i, box in enumerate(boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i,j] = two_boxes_iou(box,gt_box)
            pass
        pass
    return ious
    pass


def two_boxes_iou(box1, box2):
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)

    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max((int_x1 - int_x0 + 1),0) * max((int_y1 - int_y0 + 1),0)

    b1_area = max((b1_x1 - b1_x0 + 1),0) * max((b1_y1 - b1_y0 + 1),0)
    b2_area = max((b2_x1 - b2_x0 + 1),0) * max((b2_y1 - b2_y0 + 1),0)

    # 分母加个1e-05，避免除数为 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou
    pass

def generate_anchors(params,base_size=16):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    ratios = params.ratios
    scales = params.scales
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))
    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors
    pass

def anchors_for_shape(image_shape,anchor_params = None):
    if anchor_params is None:
        params = collections.namedtuple('AnchorParameters', ['sizes', 'strides', 'ratios', 'scales'])
        params.sizes = [32, 64, 128, 256, 512]#主干网络一共输出5个tensor，每个tensor对应一套生成anchor的size和strides参数，故而，sizes和strides都有5个元素
        params.strides = [8, 16, 32, 64, 128]
        params.ratios = np.array([0.5, 1, 2], dtype=np.float32)
        params.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=np.float32)
        anchor_params = params
        pass

    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + stride - 1) // stride for stride in anchor_params.strides]

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, _ in enumerate(anchor_params.strides):
        anchors = generate_anchors(anchor_params,base_size=anchor_params.sizes[idx])
        #shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shift(image_shapes[idx], anchor_params.strides[idx], anchors), axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0] #9
    K = shifts.shape[0] #784
    temp1 = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    temp2 = anchors.reshape((1, A, 4))
    all_anchors = (temp2 + temp1)
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors
    pass

def create_anchors_tensor(height, width, feat_stride=8, size=32, scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], ratios=[2.0, 1.0 , 0.5]):
    anchors_wh = []
    for r in ratios:
        for s in scales:
            area = np.square(s*size)
            h = np.sqrt(area/r)
            w = r * h
            anchors_wh.append([w,h])
            pass
        pass
    anchors_wh = np.array(anchors_wh)

    grid_x = np.arange(0,width) * feat_stride + feat_stride / 2
    grid_y = np.arange(0,height) * feat_stride + feat_stride / 2
    offset_x, offset_y = np.meshgrid(grid_x, grid_y)
    offset_x = np.reshape(offset_x, (-1, 1))
    offset_y = np.reshape(offset_y, (-1, 1))

    offset_xy = np.concatenate([offset_x, offset_y], -1)
    offset_xy = np.tile(offset_xy,(1,9))
    offset_xy = np.reshape(offset_xy,(-1,9,2))
    anchors_wh = np.tile(anchors_wh,(height * width,1))
    anchors_wh = np.reshape(anchors_wh, (-1, 9, 2))
    anchors_xywh = np.concatenate([offset_xy,anchors_wh],-1)
    anchors_xywh = np.expand_dims(anchors_xywh,0)
    anchors_tensor = np.tile(anchors_xywh,(1,1,1,1))
    anchors_tensor = np.reshape(anchors_tensor,(height * width * 9,4))
    anchors_x1 = anchors_tensor[:,0] - np.round(0.5 * anchors_tensor[:,2])
    anchors_y1 = anchors_tensor[:, 1] - np.round(0.5 * anchors_tensor[:, 3])
    anchors_x2 = anchors_tensor[:, 0] + np.round(0.5 * anchors_tensor[:, 2])
    anchors_y2 = anchors_tensor[:, 1] + np.round(0.5 * anchors_tensor[:, 3])

    anchors_tensor = np.concatenate([anchors_x1[:,np.newaxis],anchors_y1[:,np.newaxis],anchors_x2[:,np.newaxis],anchors_y2[:,np.newaxis]],axis=-1)

    return anchors_tensor

if __name__ == "__main__":
    # params = collections.namedtuple('AnchorParameters', ['sizes', 'strides', 'ratios', 'scales'])
    # params.sizes = [32, 64, 128, 256, 512]
    # params.strides = [8, 16, 32, 64, 128]
    # params.ratios = np.array([0.5, 1, 2], dtype=np.float32)
    # params.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=np.float32)
    # anchors = generate_anchors(params)
    #all_anchors = anchors_for_shape((224,224,3))
    #print(len(all_anchors))
    params = collections.namedtuple('AnchorParameters', ['sizes', 'strides', 'ratios', 'scales'])
    params.sizes = [32, 64, 128, 256, 512]  # 主干网络一共输出5个tensor，每个tensor对应一套生成anchor的size和strides参数，故而，sizes和strides都有5个元素
    params.strides = [8, 16, 32, 64, 128]
    params.ratios = np.array([0.5, 1, 2], dtype=np.float32)
    params.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=np.float32)
    anchor_params = params
    anchors = generate_anchors(anchor_params, base_size=anchor_params.sizes[0])
    shifted_anchors1 = shift((28,28,512), anchor_params.strides[0], anchors)
    shifted_anchors2 = create_anchors_tensor(28,28)

    pass