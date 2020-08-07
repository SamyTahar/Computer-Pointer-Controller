import cv2
import numpy as np
from model import Model
import math as m


class gazeEstimation():

    def __init__(self, MODEL_PATH, DEVICE):
        
        self.model_loaded = Model(MODEL_PATH, DEVICE)
        self.model_loaded.get_unsupported_layer()

        self.left_eyes_frame = None
        self.right_eyes_frame = None
        self.head_pose_angle = None
        self.initial_w = None
        self.initial_h = None
        self.coords = None
        self.frame = None
        self.left_eye_center = None
        self.right_eye_center = None

        self.image_input_shape = self.model_loaded.get_input_shape()
    
        
    def set_params(self, left_eyes_frame, right_eyes_frame, head_pose_angles):

        self.left_eyes_frame = left_eyes_frame
        self.right_eyes_frame = right_eyes_frame
        self.head_pose_angles = head_pose_angles

    def get_inference_outputs(self):

        
        pitch = self.head_pose_angles['angle_p_fc'][0][0]
        roll = self.head_pose_angles['angle_r_fc'][0][0]
        yaw = self.head_pose_angles['angle_y_fc'][0][0] 

        head_pose_angle = np.array((pitch,roll,yaw))

        inputs = self.input_blobs()
        
        prepro_img_left_eyes = self.preprocess_frame(self.left_eyes_frame)
        prepro_img_right_eyes = self.preprocess_frame(self.right_eyes_frame)
        
        inputs_model = {inputs[0]:head_pose_angle, inputs[1]:prepro_img_left_eyes, inputs[2]:prepro_img_right_eyes}
        
        gaze_vectors = self.inference(inputs_model)
        
        return gaze_vectors['gaze_vector'][0] 
    
    def input_blobs(self):
        return self.model_loaded.get_input_blob()

    def output_blobs(self):
        return self.model_loaded.get_output_blob()    
    
    def preprocess_frame(self, frame):

        resize_frame = cv2.resize(frame, (self.image_input_shape[1][3], self.image_input_shape[1][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        
        return self.model_loaded.get_infer_output(input_data)
        
