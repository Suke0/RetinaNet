#-- coding: utf-8 --
import tensorflow as tf
import numpy as np

def res_net50(x):
    x = conv_stage_1(x)
    x = conv_stage_2(x)
    x = conv_stage_3(x)
    c3 = x
    x = conv_stage_4_(x)
    c4 = x
    x = conv_stage_5(x)
    c5 = x
    return c3, c4, c5  #(bacthsize, 28, 28, 512) (bacthsize, 14, 14, 1024) (bacthsize, 7, 7, 2048)
    pass


def res_net101(x):
    x = conv_stage_1(x)
    x = conv_stage_2(x)
    x = conv_stage_3(x)
    c3 = x
    x = conv_stage_4(x)
    c4 = x
    x = conv_stage_5(x)
    c5 = x
    return c3, c4, c5  #(bacthsize, 28, 28, 512) (bacthsize, 14, 14, 1024) (bacthsize, 7, 7, 2048)
    pass


def conv_stage_1(x):#224
    x = conv_bn_relu(64, 7, 2,padding="same")(x)
    return x   #112
    pass


def conv_stage_2(x):#112
    x = tf.keras.layers.MaxPooling2D(3, 2, padding="same")(x)
    # 由于上下组的卷积层通道数不同，使得短路连接不能直接相加，故需要在后四组连接上一组的第一个卷积层的短路连接通路添加投影卷积。
    x_ = conv_bn(64 * 4, 1, 1)(x)
    x_in = conv_bn_relu(64, 1, 1)(x)
    x_in = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    x_in = conv_bn_relu(64, 3, 1)(x_in)
    x_in = conv_bn(64 * 4, 1, 1)(x_in)
    x_in = tf.keras.layers.Add()([x_in, x_])
    x_in = tf.keras.layers.Activation("relu")(x_in)
    for i in range(2):
        x_in = res_block(x_in, 64, 1)
    return x_in   #56
    pass


def conv_stage_3(x):#56
    # 由于上下组的卷积层通道数不同，使得短路连接不能直接相加，故需要在后四组连接上一组的第一个卷积层的短路连接通路添加投影卷积。
    x_ = conv_bn(128 * 4, 1, 2)(x)
    x_in = conv_bn_relu(128, 1, 2)(x)
    x_in = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    x_in = conv_bn_relu(128, 3, 1)(x_in)
    x_in = conv_bn(128 * 4, 1, 1)(x_in)
    x_in = tf.keras.layers.Add()([x_in, x_])
    x_in = tf.keras.layers.Activation("relu")(x_in)

    for i in range(3):
        x_in = res_block(x_in, 128, 1)
    return x_in  #28
    pass

def conv_stage_4_(x):#28
    # 由于上下组的卷积层通道数不同，使得短路连接不能直接相加，故需要在后四组连接上一组的第一个卷积层的短路连接通路添加投影卷积。
    x_ = conv_bn(256 * 4, 1, 2)(x)
    x_in = conv_bn_relu(256, 1, 2)(x)
    x_in = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    x_in = conv_bn_relu(256, 3, 1)(x_in)
    x_in = conv_bn(256 * 4, 1, 1)(x_in)
    x_in = tf.keras.layers.Add()([x_in, x_])
    x_in = tf.keras.layers.Activation("relu")(x_in)
    for i in range(5):
        x_in = res_block(x_in, 256, 1)
    return x_in  #14
    pass

def conv_stage_4(x):#28
    # 由于上下组的卷积层通道数不同，使得短路连接不能直接相加，故需要在后四组连接上一组的第一个卷积层的短路连接通路添加投影卷积。
    x_ = conv_bn(256 * 4, 1, 2)(x)
    x_in = conv_bn_relu(256, 1, 2)(x)
    x_in = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    x_in = conv_bn_relu(256, 3, 1)(x_in)
    x_in = conv_bn(256 * 4, 1, 1)(x_in)
    x_in = tf.keras.layers.Add()([x_in, x_])
    x_in = tf.keras.layers.Activation("relu")(x_in)
    for i in range(22):
        x_in = res_block(x_in, 256, 1)
    return x_in  #14
    pass


def conv_stage_5(x):#14
    #由于上下组的卷积层通道数不同，使得短路连接不能直接相加，故需要在后四组连接上一组的第一个卷积层的短路连接通路添加投影卷积。
    x_ = conv_bn(512 * 4, 1, 2)(x)
    x_in = conv_bn_relu(512, 1, 2)(x)
    x_in = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    x_in = conv_bn_relu(512, 3, 1)(x_in)
    x_in = conv_bn(512 * 4, 1, 1)(x_in)
    x_in = tf.keras.layers.Add()([x_in, x_])
    x_in = tf.keras.layers.Activation("relu")(x_in)
    for i in range(2):
        x_in = res_block(x_in, 512, 1)
    return x_in  #7
    pass


def res_block(x, filters,strides):
    x_in = conv_bn_relu(filters, 1, strides)(x)
    x_in = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    x_in = conv_bn_relu(filters, 3, strides)(x_in)
    x_in = conv_bn(filters * 4, 1, strides)(x_in)
    x_in = tf.keras.layers.Add()([x_in, x])
    x_in = tf.keras.layers.Activation("relu")(x_in)
    return x_in
    pass


def conv_bn_relu(filters, filter_size,strides,padding='valid'):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, filter_size, strides, padding=padding,use_bias= False),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.Activation("relu")
    ])

def conv_bn(filters, filter_size, strides,padding='valid'):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, filter_size, strides, padding=padding,use_bias= False),
        tf.keras.layers.BatchNormalization(epsilon=1e-5)
    ])
    pass


def ResNet2D50(inputs, blocks=None):
    if blocks is None:
        blocks = [3, 4, 6, 3]
        pass

    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    features = 64
    outputs = []
    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = bottleneck(features, stage_id, block_id)(x)
            pass
        features *= 2
        outputs.append(x)
        pass
    return outputs[1:]
    pass

def bottleneck(filters,stage=0,block=0,kernel_size=3,stride=None):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2
    def f(x):
        y = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False)(x)

        y = tf.keras.layers.BatchNormalization(epsilon=1e-5)(y)

        y = tf.keras.layers.Activation("relu")(y)

        y = tf.keras.layers.ZeroPadding2D(padding=1)(y)

        y = tf.keras.layers.Conv2D(filters, kernel_size, use_bias=False)(y)

        y = tf.keras.layers.BatchNormalization(epsilon=1e-5)(y)

        y = tf.keras.layers.Activation("relu")(y)

        y = tf.keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False)(y)

        y = tf.keras.layers.BatchNormalization(epsilon=1e-5)(y)

        if block == 0:
            shortcut = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False)(x)

            shortcut = tf.keras.layers.BatchNormalization(epsilon=1e-5)(shortcut)
        else:
            shortcut = x
        y = tf.keras.layers.Add()([y, shortcut])
        y = tf.keras.layers.Activation("relu")(y)
        return y
        pass
    return f
    pass


if __name__ == '__main__':
    input =tf.constant(np.random.rand(2,224,224,3),dtype=tf.float32)
    res = res_net101(input)
    print(res[0].shape,res[1].shape,res[2].shape) #(bacthsize, 28, 28, 512) (bacthsize, 14, 14, 1024) (bacthsize, 7, 7, 2048)
    pass