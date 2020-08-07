import cv2
import numpy as np
from model import Model
import math as m
import utils


class FaceDetection():

    def __init__(self, MODEL_PATH, DEVICE):
        
        self.model_loaded = Model(MODEL_PATH, DEVICE)
        self.model_loaded.get_unsupported_layer()

        self.frame = None
        self.initial_w = None
        self.initial_h = None
        self.coords = None
        self.threshold = None

        self.image_input_shape = self.model_loaded.get_input_shape()
    
        
    def set_params(self, frame, threshold, initial_w, initial_h):
        self.frame = frame
        self.initial_w = initial_w
        self.initial_h = initial_h
        self.threshold = threshold

    def get_inference_outputs(self):

        inputs_facedetect = self.input_blobs()
        prepro_img_face = self.preprocess_frame(self.frame)
        inputs_face = {inputs_facedetect[0]:prepro_img_face}
        coords = self.inference(inputs_face)

        confidence, data_face_detection  = self.get_box_data(coords, self.threshold, self.initial_w, self.initial_h)
        
        return confidence, data_face_detection
    
    def input_blobs(self):
        return self.model_loaded.get_input_blob()

    def output_blobs(self):
        return self.model_loaded.get_output_blob()    
    
    def preprocess_frame(self, frame):
        resize_frame = cv2.resize(frame, (self.image_input_shape[0][3], self.image_input_shape[0][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        
        return self.model_loaded.get_infer_output(input_data)

    def get_box_data(self, coords, threshold , initial_w, initial_h):
        width = initial_w
        height = initial_h
        coords = coords['detection_out']
        data_face_detection = []
        _confidence = 0
            
        for box in coords[0][0]: # Output shape is 1x1x200x7
            confidence = box[2]
          
            if confidence >= threshold:
    
                xmin = int(box[3] * width) - 50
                ymin = int(box[4] * height) - 50
                xmax = int(box[5] * width) + 50
                ymax = int(box[6] * height) + 50

                _confidence = confidence
                #coords_data_crop = ymin, ymax, xmin, xmax
                data_face_detection = xmin, ymin, xmax, ymax

        return _confidence, data_face_detection

        
