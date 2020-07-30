import cv2
import numpy as np
from model import Model


class gazeEstimation():

    def __init__(self, MODEL_PATH):
        #super().__init__()
        self.model_loaded = Model(MODEL_PATH)
        self.model_loaded.get_unsupported_layer()
        #model_name =  self.model_loaded.get_model_name()
        #print("log[info]: Model input shape " + model_name +" ", self.model_loaded.get_input_shape())
        #print("log[info]: Model output shape " + model_name +" ", self.model_loaded.get_output_shape())

        self.left_eyes_frame = None
        self.right_eyes_frame = None
        self.head_pose_angle = None
        self.initial_w = None
        self.initial_h = None
        self.coords = None

        #print("input_names:", inputs,  "inputs_shape: ",inputs_shape)
        #print("output_names:", outputs, "outputs_shape: ",outputs_shape)
    
        
    def set_params(self, left_eyes_frame, right_eyes_frame, head_pose_angle, initial_w, initial_h):
        
        head_pose_angle = np.array((head_pose_angle['angle_p_fc'][0][0], head_pose_angle['angle_r_fc'][0][0], head_pose_angle['angle_y_fc'][0][0]))
        print(head_pose_angle)      
        self.left_eyes_frame = left_eyes_frame
        self.right_eyes_frame = right_eyes_frame
        self.head_pose_angle = head_pose_angle

        self.initial_w = initial_w
        self.initial_h = initial_h

    def get_inference_outputs(self):

        inputs = self.input_blobs()
        prepro_img_left_eyes = self.preprocess_frame(self.left_eyes_frame)
        prepro_img_right_eyes = self.preprocess_frame(self.right_eyes_frame)
        inputs_model = {inputs[0]:self.head_pose_angle, inputs[1]:prepro_img_left_eyes, inputs[2]:prepro_img_right_eyes}
        gaze_vectors = self.inference(inputs_model)
        
        return gaze_vectors
    
    def input_blobs(self):
        return self.model_loaded.get_input_blob()

    def output_blobs(self):
        return self.model_loaded.get_output_blob()    
    
    def preprocess_frame(self, frame):
        
        image_input_shape = self.model_loaded.get_input_shape()

        resize_frame = cv2.resize(frame, (image_input_shape[1][3], image_input_shape[1][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        
        return self.model_loaded.get_infer_output(input_data)    
    
    def crop_image(self,img, y1,y2,x1,x2):
        
        return img[y1:y2, x1:x2]
         
