# -*- coding: utf-8 -*-

import os
import sys

import caffe
import cv2
import numpy as np

from feature_extractor import FeatureExtractor


def load_keys(path):
    keys = []
    for line in open(path):
        line = line.strip()
        keys.append(line)
    return keys


def convert_mean_file(path):
    data = open(os.path.join(path, "imagenet_mean.binaryproto"), "rb").read()
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)
    mean = np.asarray(caffe.io.blobproto_to_array(blob))[0]
    np.save(os.path.join(path, "imagenet_mean.npy"), mean)


def main():
    caffe_alexnet_path = "/mnt/hdd1/projects/caffe-modelzoo/AlexNet"
    caffe_vgg16_path = "/mnt/hdd1/projects/caffe-modelzoo/VGG16"
    caffe_googlenet_path = "/mnt/hdd1/projects/caffe-modelzoo/GoogleNet"
    keys_path = "/mnt/hdd1/dataset/PascalSentence/keys.txt"
    data_path = "/mnt/hdd1/dataset/PascalSentence/images"
    dst_path = "/mnt/hdd1/projects/image-feature-extraction-via-convnet/features.npy"

    modelname = "VGG16"

    # load pre-trained model
    if modelname == "AlexNet":
        if not os.path.exists(os.path.join(caffe_alexnet_path, "imagenet_mean.npy")):
            convert_mean_file(caffe_alexnet_path)
        convnet = FeatureExtractor(
                prototxt_path=os.path.join(caffe_alexnet_path, "alexnet_deploy.prototxt"),
                caffemodel_path=os.path.join(caffe_alexnet_path, "alexnet.caffemodel"),
                target_layer_name="fc6",
                image_size=227,
                mean_path=os.path.join(caffe_alexnet_path, "imagenet_mean.npy")
                )
    elif modelname == "VGG16":
        convnet = FeatureExtractor(
                prototxt_path=os.path.join(caffe_vgg16_path, "vgg16_deploy.prototxt"),
                caffemodel_path=os.path.join(caffe_vgg16_path, "vgg16.caffemodel"),
                target_layer_name="fc6",
                image_size=224,
                mean_values=[103.939, 116.779, 123.68]
                )
    elif modelname == "GoogleNet":
        googlenet = FeatureExtractor(
                prototxt_path=os.path.join(caffe_googlenet_path, "googlenet_deploy.prototxt"),
                caffemodel_path=os.path.join(caffe_googlenet_path, "googlenet.caffemodel"),
                target_layer_name="pool5/7x7_s1",
                image_size=224,
                mean_values=[104.0, 117.0, 123.0]
                )
    else:
        print "Unknown model name: %s" % modelname
        sys.exit(-1)
    
    # data list
    keys = load_keys(keys_path)
    
    # feature extraction
    feats = []
    for key in keys:
        img = cv2.imread(os.path.join(data_path, key))
        assert img is not None
        feat = convnet.transform(img)
        feats.append(feat)
    feats = np.asarray(feats)
    np.save(dst_path, feats)

    print "Done."

if __name__ == "__main__":
    main()

