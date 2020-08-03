import cv2
import numpy as np
from model import Model
import math as m


class headPoseEstimation():

    def __init__(self, MODEL_PATH, DEVICE):
        
        self.model_loaded = Model(MODEL_PATH, DEVICE)
        self.model_loaded.get_unsupported_layer()

        self.initial_w = None
        self.initial_h = None
        self.frame = None

        self.image_input_shape = self.model_loaded.get_input_shape()

    def input_blobs(self):

        print(self.model_loaded.get_input_blob())
        return self.model_loaded.get_input_blob()

    def output_blobs(self):

        print(self.model_loaded.get_output_blob())
        return self.model_loaded.get_output_blob() 

    def set_params(self, frame, initial_w, initial_h):

        self.frame = frame
        self.initial_w = initial_w
        self.initial_h = initial_h

    def get_inference_outputs(self):
       
        inputs_model = self.input_blobs()
        prepro_img_face = self.preprocess_frame(self.frame)
        inputs_to_feed = {inputs_model[0]:prepro_img_face}    
        angles = self.inference(inputs_to_feed)

        frame = self.display_head_orientation(self.frame, angles, self.initial_w, self.initial_h)

        return frame ,angles 
    
    def preprocess_frame(self, frame):

        resize_frame = cv2.resize(frame, (self.image_input_shape[0][3], self.image_input_shape[0][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        return self.model_loaded.get_infer_output(input_data)
    
    def display_head_orientation(self, frame, eurler_angles,initial_w, initial_h):
            
        pitch = eurler_angles['angle_p_fc'][0][0]
        roll  = eurler_angles['angle_r_fc'][0][0]
        yaw = eurler_angles['angle_y_fc'][0][0]
        

        #x = m.cos(yaw)*m.cos(pitch)
        #y = m.sin(yaw)*m.cos(pitch)
        #z = m.sin(pitch)
        center_point = [(initial_w+0)/2,(initial_h+0)/2]
        #print(x,y,z)

        print(center_point)

        #line_thickness = 5
        #frame = cv2.line(frame, (int(center_point[0]),int(center_point[1])), (int(center_point[0]), 300), (0, 255, 0), thickness=line_thickness)
        #rame = cv2.line(frame, (int(center_point[0]),int(center_point[1])), (700, int(center_point[1])), (0, 0, 255), thickness=line_thickness)
        #frame = cv2.line(frame, (int(center_point[0]),int(center_point[1])), (730, 450), (255, 0, 0), thickness=line_thickness)

        #frame = cv2.line(frame, (x, y), (x, z), (0, 255, 0), thickness=line_thickness)
        
        return frame

