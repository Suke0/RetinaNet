# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 23:25
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : loss.py
# @Software: PyCharm
import tensorflow as tf
def focal(y_true, y_pred, alpha=0.25, gamma=2.0):
    labels = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1] # -1 for ignore, 0 for background, 1 for object
    pre_cls = y_pred

    #filter out ignore anchors
    indices = tf.where(tf.math.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    pre_cls = tf.gather_nd(pre_cls,indices)

    #compute the focal loss
    alpha_factor = tf.ones_like(labels) * alpha
    alpha_factor = tf.where(tf.math.equal(labels,1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.math.equal(labels,1), 1 - pre_cls, pre_cls)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(labels,pre_cls)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(tf.math.equal(anchor_state, 1))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.math.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

    return tf.keras.backend.sum(cls_loss) / normalizer
    pass

def smooth_l1(y_true, y_pred,sigma=3.0):
    sigma_squared = sigma ** 2
    # separate target and state
    regression = y_pred
    regression_target = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]

    # filter out "ignore" anchors
    indices = tf.where(tf.math.equal(anchor_state, 1))
    regression = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = tf.math.abs(regression_diff)
    regression_loss = tf.where(tf.math.less(regression_diff, 1.0 / sigma_squared),
                               0.5 * sigma_squared * tf.math.pow(regression_diff, 2),
                               regression_diff - 0.5 / sigma_squared)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.math.maximum(1, tf.shape(indices)[0])
    normalizer = tf.cast(normalizer, dtype=tf.float32)
    return tf.keras.backend.sum(regression_loss) / normalizer
    pass