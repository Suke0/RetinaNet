# -*- coding: utf-8 -*-
# @Time    : 2020/3/6 2:08
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : generator.py
# @Software: PyCharm
import tensorflow as tf
import math
import numpy as np
import os
from xml.etree.ElementTree import parse
from random import shuffle
from PIL import Image
from anchor import create_anchor_targets
from image import TransformParameters, adjust_transform_for_image, apply_transform, resize_image, random_visual_effect_generator
from transform import transform_aabb, random_transform_generator


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, ann_fnames, img_dir, label_names, batch_size, image_min_side=800, image_max_side=1333, jitter=True, shuffle=True):
        self.ann_fnames = ann_fnames
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_classes = len(label_names)
        self.jitter = jitter
        self.transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        self.visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        ) #生成从给定间隔均匀采样的视觉效果参数
        self.transform_parameters = TransformParameters()
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.shuffle = shuffle
        self.label_names = label_names
        self.create_anchor_targets = create_anchor_targets
        pass

    def __len__(self):
        return math.ceil(len(self.ann_fnames) / self.batch_size)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        inputs, targets = self.compute_input_output(index)
        #print("---------------"+str(inputs[0].shape))
        #print("---------------" + str(index))
        #print("---------------" + str(targets[0].shape))
        return inputs, targets
        pass

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.ann_fnames)
            pass
        pass

    def filter_annotations(self, image_group, box_group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, gt_boxes) in enumerate(zip(image_group, box_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (gt_boxes[:, 2] <= gt_boxes[:, 0]) |
                (gt_boxes[:, 3] <= gt_boxes[:, 1]) |
                (gt_boxes[:, 0] < 0) |
                (gt_boxes[:, 1] < 0) |
                (gt_boxes[:, 2] > image.shape[1]) |
                (gt_boxes[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                for k in box_group[index].keys():
                    box_group[index][k] = np.delete(gt_boxes[k], invalid_indices, axis=0)
        return image_group, box_group

    def random_visual_effect_group_entry(self, image, gt_boxes):
        """ Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, gt_boxes
        pass

    def random_visual_effect_group(self, image_group, box_group):
        """ Randomly apply visual effect on each image.
        """
        assert (len(image_group) == len(box_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, box_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], box_group[index] = self.random_visual_effect_group_entry(
                image_group[index], box_group[index]
            )

        return image_group, box_group
        pass

    def random_transform_group_entry(self, image, gt_boxes, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image,self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            gt_boxes = gt_boxes.copy()
            for index in range(gt_boxes.shape[0]):

                gt_boxes[index, :4] = transform_aabb(transform, gt_boxes[index, :4])

        return image, gt_boxes

    def random_transform_group(self, image_group, box_group):
        """ Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(box_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], box_group[index] = self.random_transform_group_entry(image_group[index], box_group[index])

        return image_group, box_group
        pass

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)
        pass

    def preprocess_group_entry(self, image, gt_boxes):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        #image = self.preprocess_image(image)

        image = image.astype(np.float32)
        image /= 127.5
        image -= 1.

        # resize image
        image, image_scale = self.resize_image(image)
        gt_boxes = gt_boxes.astype(np.float32)
        # apply resizing to annotations too
        gt_boxes[:,:4] *= image_scale

        # convert to the wanted keras floatx
        image = tf.keras.backend.cast_to_floatx(image)

        return image, gt_boxes
        pass

    def preprocess_group(self, image_group, box_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(box_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], box_group[index] = self.preprocess_group_entry(image_group[index],box_group[index])

        return image_group, box_group
        pass

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=tf.keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch
        pass

    def compute_targets(self, image_group, box_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        batches = self.create_anchor_targets(image_group,box_group,self.num_classes)

        return list(batches)
        pass

    def compute_input_output(self,index):
        image_group = []
        box_group = []
        for i in range(self.batch_size):
            fname, gt_boxes = parse_annotation(self.ann_fnames[index * self.batch_size + i], self.img_dir, self.label_names)

            # 读取图片
            image = np.asarray(Image.open(fname).convert('RGB'))
            image = image[:, :, ::-1].copy()
            image_group.append(image)
            box_group.append(gt_boxes)
            pass
        #image_group, box_group = np.array(image_group), np.array(box_group)
        # check validity of annotations
        image_group, box_group = self.filter_annotations(image_group, box_group)

        # randomly apply visual effect
        image_group, box_group = self.random_visual_effect_group(image_group, box_group)

        # randomly transform data
        image_group, box_group = self.random_transform_group(image_group, box_group)

        # perform preprocessing steps
        image_group, box_group = self.preprocess_group(image_group, box_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, box_group)
        return inputs, targets
        pass

    pass

class PascalVocXmlParser(object):
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels

    def get_boxes(self, annotation_file):
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree

    pass

class Annotation(object):
    def __init__(self, filename):
        self.fname = filename
        #self.labels = []
        #self.coded_labels = []
        self.boxes = None
        pass

    def add_object(self, x1, y1, x2, y2, code):
        #self.labels.append(name)
        #self.coded_labels.append(code)
        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2, code]).reshape(-1, 5)
        else:
            box = np.array([x1, y1, x2, y2, code],dtype = np.float32).reshape(-1, 5)
            self.boxes = np.concatenate([self.boxes, box])
        pass

    pass

def parse_annotation(ann_fname, img_dir, labels_name=[]):
    parser = PascalVocXmlParser()
    fname = parser.get_fname(ann_fname)

    annotation = Annotation(os.path.join(img_dir, fname))

    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        if label in labels_name:
            annotation.add_object(x1, y1, x2, y2, labels_name.index(label))
    return annotation.fname, annotation.boxes
    pass
