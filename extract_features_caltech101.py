# -*- coding: utf-8 -*-

import os
import sys

import cv2
import pyprind

from feature_extractor import FeatureExtractor


def main():
    dataset_path = "/path/to/Caltech-101"
    modelzoo_path = "/path/to/VGG16"
    
    # create an instance
    convnet = FeatureExtractor(
            prototxt_path=os.path.join(modelzoo_path, "vgg16_deploy.prototxt"),
            caffemodel_path=os.path.join(modelzoo_path, "vgg16.caffemodel"),
            target_layer_name="fc7",
            image_size=224,
            mean_values=[103.939, 116.779, 123.68])
    
    # header
    f = open("caltech101_vggnet_fc7_features.csv", "w")
    header = ["filepath"]
    for i in xrange(4096):
        header.append("feat%d" % (i+1))
    header = ",".join(header) + "\n"
    f.write(header)
    
    # extract features
    categories = os.listdir(dataset_path)
    for category in pyprind.prog_bar(categories):
        file_names = os.listdir(os.path.join(dataset_path, category))
        for file_name in file_names:
            img = cv2.imread(os.path.join(dataset_path, category, file_name))
            feat = convnet.transform(img)
            feat_str = [os.path.join(category, file_name)]
            for value in feat:
                feat_str.append(str(value))
            row = ",".join(feat_str)
            f.write("%s\n" % row)
            f.flush()

    f.close()


if __name__ == "__main__":
    main()
            



