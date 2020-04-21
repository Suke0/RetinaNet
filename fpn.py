#-- coding: utf-8 --
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def fpn(c3,c4,c5,filters=256):
    # upsample C5 to get P5 from the FPN paper
    p5 = tf.keras.layers.Conv2D(filters,1,1,padding='same')(c5)
    p5_ = UpsampleLike()([p5, c4])
    p5 = tf.keras.layers.Conv2D(filters,3,1,padding='same')(p5)

    # add P5 elementwise to C4
    p4 = tf.keras.layers.Conv2D(filters, 1, 1, padding='same')(c4)
    p4 = tf.keras.layers.Add()([p5_, p4])
    p4_ = UpsampleLike()([p4, c3])
    p4 = tf.keras.layers.Conv2D(filters, 3, 1, padding='same')(p4)


    # add P4 elementwise to C3
    p3 = tf.keras.layers.Conv2D(filters, 1, 1, padding='same')(c3)
    p3 = tf.keras.layers.Add()([p4_, p3])
    p3 = tf.keras.layers.Conv2D(filters, 3, 1, padding='same')(p3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    p6 = tf.keras.layers.Conv2D(filters,3,2,padding='same')(c5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    p7 = tf.keras.layers.ReLU()(p6)
    p7 = tf.keras.layers.Conv2D(filters,3,2,padding='same')(p7)

    return [p3, p4, p5, p6, p7]
    pass

# class UpsampleLike(tf.keras.layers.Layer):
#     """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
#     """
#     def call(self, inputs,shape, **kwargs):
#         return tf.keras.backend.resize_images(inputs, (shape[1], shape[2]), method='nearest')
#         pass
#     pass

class UpsampleLike(tf.keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = tf.keras.backend.shape(target)
        return tf.image.resize(source, target_shape[1:3], method=ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
        pass
