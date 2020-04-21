#-- coding: utf-8 --
import tensorflow as tf
import numpy as np
import math
# 在p3-p7层上选用的anchors拥有的像素区域大小从32x32到512x512,每层之间的长度是两倍的关系。
# 每个金字塔层有三种长宽比例[1:2 ,1:1 ,2:1]，有三种尺寸大小[2^0, 2^（1/3)， 2^（2/3)]。
# 总共便是每层9个anchors。大小从32像素到813像素。 32 = 32 * 2^0, 813 = 512 * 2^（2/3)

# 分类子网络和回归子网络的参数是分开的，但结构却相似。都是用小型FCN网络，
# 将金字塔层作为输入，接着连接4个3x3的卷积层，fliter为金字塔层的通道数（论文中是256)，
# 每个卷积层后都有RELU激活函数，这之后连接的是fliter为KA
# （K是目标种类数，A是每层的anchors数，论文中是9)的3x3的卷积层，激活函数是sigmoid。


# def detectnet(input, num_classes, filters=256, n_anchors=9):
#     #res_array = []
#     res_cls = []
#     res_bbox = []
#     for x in list:
#         x_cls = conv_bn_relu(filters, 3, 1)(x)
#         x_cls = conv_bn_relu(filters, 3, 1)(x_cls)
#         x_cls = conv_bn_relu(filters, 3, 1)(x_cls)
#         x_cls = conv_bn_relu(filters, 3, 1)(x_cls)
#         x_cls = tf.keras.layers.Conv2D(n_anchors * num_classes, 3, 1, padding='same')(x_cls)
#         x_cls = tf.keras.layers.Activation("sigmoid")(x_cls)
#         print(x_cls.shape)
#         #res_array.append(x_cls)
#         res_cls.append(x_cls)
#         pass
#     for x_ in list:
#         x_bbox = conv_bn_relu(filters, 3, 1)(x_)
#         x_bbox = conv_bn_relu(filters, 3, 1)(x_bbox)
#         x_bbox = conv_bn_relu(filters, 3, 1)(x_bbox)
#         x_bbox = conv_bn_relu(filters, 3, 1)(x_bbox)
#         x_bbox = tf.keras.layers.Conv2D(n_anchors * 4, 3, 1, padding='same')(x_bbox)
#         x_bbox = tf.keras.layers.Activation("sigmoid")(x_bbox)
#         #print(x_bbox.shape)
#         #res_array.append(x_bbox)
#         res_bbox.append(x_bbox)
#         pass
#
#     return res_cls,res_bbox
#     pass

def detectnet(list,num_classes):
    reg_model = create_reg_model()
    cls_model = create_cls_model(num_classes)
    #res_array = []
    res_cls = []
    res_bbox = []

    for x in list:
        x_bbox = reg_model(x) #list里的元素检测时，卷积核参数共享
        #print(x_bbox.shape)
        #res_array.append(x_bbox)
        res_bbox.append(x_bbox)

        x_cls = cls_model(x)  # list里的元素检测时，卷积核参数共享
        # print(x_cls.shape)
        # res_array.append(x_cls)
        res_cls.append(x_cls)

        pass

    # for x in list:
    #     x_cls = cls_model(x)  # list里的元素检测时，卷积核参数共享
    #     # print(x_cls.shape)
    #     # res_array.append(x_cls)
    #     res_cls.append(x_cls)
    #     pass

    # (2, 7056, 20)
    # (2, 1764, 20)
    # (2, 441, 20)
    # (2, 144, 20)
    # (2, 36, 20)
    # (2, 7056, 4)
    # (2, 1764, 4)
    # (2, 441, 4)
    # (2, 144, 4)
    # (2, 36, 4)

    res_bbox = tf.concat(res_bbox, axis=1)
    res_cls = tf.concat(res_cls,axis=1)


    return res_bbox, res_cls
    pass

def create_reg_model(filters=256, n_anchors=9):
    inputs = tf.keras.Input(shape=(None, None, filters))
    x_bbox = conv_relu(filters, 3, 1)(inputs)
    x_bbox = conv_relu(filters, 3, 1)(x_bbox)
    x_bbox = conv_relu(filters, 3, 1)(x_bbox)
    x_bbox = conv_relu(filters, 3, 1)(x_bbox)
    x_bbox = tf.keras.layers.Conv2D(n_anchors * 4, 3, 1, padding='same')(x_bbox)
    x_bbox = tf.keras.layers.Reshape((-1, 4))(x_bbox)
    #x_bbox = tf.keras.layers.Activation("sigmoid")(x_bbox)
    return tf.keras.Model(inputs=inputs,outputs=x_bbox)
    pass


def create_cls_model(num_classes, filters=256, n_anchors=9):
    inputs = tf.keras.Input(shape=(None, None, filters))
    x_cls = conv_relu(filters, 3, 1)(inputs)
    x_cls = conv_relu(filters, 3, 1)(x_cls)
    x_cls = conv_relu(filters, 3, 1)(x_cls)
    x_cls = conv_relu(filters, 3, 1)(x_cls)
    x_cls = tf.keras.layers.Conv2D(n_anchors * num_classes, 3, 1, padding='same',bias_initializer= PriorProbability(probability=0.01))(x_cls)
    x_cls = tf.keras.layers.Reshape((-1,num_classes))(x_cls)
    x_cls = tf.keras.layers.Activation("sigmoid")(x_cls)
    return tf.keras.Model(inputs=inputs,outputs=x_cls)
    pass

def conv_relu(filters, filter_size,strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, filter_size, strides, padding='same'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    pass

#分类分支的最后一级卷积的bias初始化
class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape) * -math.log((1 - self.probability) / self.probability)
        return result
    pass
