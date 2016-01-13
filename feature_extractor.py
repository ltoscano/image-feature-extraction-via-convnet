# -*- coding: utf-8 -*-


import numpy as np
import caffe
import cv2


class FeatureExtractor:
    def __init__(self,
                 prototxt_path,
                 caffemodel_path,
                 target_layer_name,
                 image_size,
                 mean_path=None,
                 mean_values=None
                 ):

        caffe.set_mode_gpu()
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.target_layer_name = target_layer_name
        self.image_size = image_size

        if mean_path is not None:
            self.mean = np.load(mean_path) # expect (1, C, H, W)
            self.mean = self.mean[0] # (C, H, W)
            self.mean = self.crop(self.mean, crop_size=self.image_size)
        elif mean_values is not None:
            assert len(mean_values) == 3 # mean values of B,G,R
            self.mean = np.zeros((3, self.image_size, self.image_size))
            self.mean[0, :, :] = mean_values[0]
            self.mean[1, :, :] = mean_values[1]
            self.mean[2, :, :] = mean_values[2]
        else:
            raise Exception

        self.create_network()


    def create_network(self):
        self.net = caffe.Net(self.prototxt_path, self.caffemodel_path, caffe.TEST)
        self.net.blobs["data"].reshape(1, 3, self.image_size, self.image_size)

    
    def crop(self, img, crop_size):
        # img.shape = (C, H, W)
        corner_size = img.shape[1] - crop_size
        corner_size = np.floor(corner_size / 2)
        res = img[:, corner_size:crop_size+corner_size, corner_size:crop_size+corner_size]
        return res
   
    
    def preprocess(self, img):
        """
        前処理
        """
        # img.shape = (H, W, C)
        # channel: BGR
        # img.dtype = np.uint8
        # value range: [0, 255]
        x = cv2.resize(img, (self.image_size, self.image_size))
        x = x.transpose((2, 0, 1))
        x = x.astype(np.float32)
        x -= self.mean
        return x 


    def transform(self, img):
        """
        特徴抽出
        """
        x = self.preprocess(img)
        hs = self.net.forward_all(**{self.net.inputs[0]: x, "blobs": [self.target_layer_name]})
        v = hs[self.target_layer_name][0]
        return v
