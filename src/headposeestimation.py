import cv2
import numpy as np
from model import Model
import math as m

import time
import logging as log


class headPoseEstimation():

    def __init__(self, MODEL_PATH, DEVICE):
        
        self.model_loaded = Model(MODEL_PATH, DEVICE)
        self.model_loaded.get_unsupported_layer()
        self.model_name = self.model_loaded.get_model_name()

        self.initial_w = None
        self.initial_h = None
        self.frame = None

        self.image_input_shape = self.model_loaded.get_input_shape()

    def input_blobs(self):
        
        return self.model_loaded.get_input_blob()

    def output_blobs(self):

        return self.model_loaded.get_output_blob() 

    def set_params(self, frame, initial_w, initial_h):

        self.frame = frame
        self.initial_w = initial_w
        self.initial_h = initial_h

    def get_inference_outputs(self):
        
        t0 = time.perf_counter()
        t_count = 0

        inputs_model = self.input_blobs()
        prepro_img_face = self.preprocess_frame(self.frame)
        inputs_to_feed = {inputs_model[0]:prepro_img_face}
        
        t_start = time.perf_counter()

        head_pose_angles = self.inference(inputs_to_feed)

        t_end = time.perf_counter()
        t_count += 1
        log.info("model {} is processed with {:0.2f} requests/sec ({:0.2} sec per request)".format(self.model_name, 1 / (t_end - t_start), t_end - t_start))

        return  head_pose_angles 
    
    def preprocess_frame(self, frame):

        resize_frame = cv2.resize(frame, (self.image_input_shape[0][3], self.image_input_shape[0][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        return self.model_loaded.get_infer_output(input_data)


