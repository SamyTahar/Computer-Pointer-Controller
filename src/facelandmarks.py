
import cv2
import numpy as np
from model import Model

import utils

class FaceLandmarks(Model):

    def __init__(self, MODEL_PATH, DEVICE):

        self.model_loaded = Model(MODEL_PATH, DEVICE)
        self.model_loaded.get_unsupported_layer()

        self.frame = None
        self.initial_w = None
        self.initial_h = None

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

        inputs_model = self.input_blobs()
        prepro_img_face = utils.preprocess_frame(utils.check_frame_shape_error(self.frame), self.image_input_shape)
        inputs_to_feed = {inputs_model[0]:prepro_img_face}
        
        points = self.inference(inputs_to_feed)

        coords_data_crop_l, coords_data_crop_r, data_points_marks = self.get_box_eyes_data(points, self.initial_h, self.initial_w)
                
        left_eye_center_points, right_eye_center_points = self.get_eyes_center(points,self.initial_h, self.initial_w)

        return left_eye_center_points ,right_eye_center_points, coords_data_crop_l, coords_data_crop_r, data_points_marks
    

    def inference(self, input_data):
        return self.model_loaded.get_infer_output(input_data)
         

    def get_box_eyes_data(self, points, frame_cropped_w, frame_cropped_h):
   
        points = points['95']

        data_points_marks= []

        #print("points:", points[0][0])
        for point in points:
            xl,yl = point[0][0] * frame_cropped_w, point[1][0] * frame_cropped_h
            xr,yr = point[2][0] * frame_cropped_w, point[3][0] * frame_cropped_h
            xn,yn = point[4][0] * frame_cropped_w, point[5][0] * frame_cropped_h

            # make box for left eye 
            xlmin = xl-50
            ylmin = yl-50
            xlmax = xl+50
            ylmax = yl+50
            
            # make box for right eye 
            xrmin = xr-50
            yrmin = yr-50
            xrmax = xr+50
            yrmax = yr+50


        coords_data_crop_l = int(ylmin), int(ylmax), int(xlmin), int(xlmax)
        coords_data_crop_r = int(yrmin), int(yrmax), int(xrmin), int(xrmax)
        data_points_marks = xl, yl, xr, yr, xn, yn 
        
        return coords_data_crop_l, coords_data_crop_r, data_points_marks

    def get_eye_frame_cropped(self,frame, coords_data_crop_l, coords_data_crop_r ):
        img_left_eye = utils.crop_image(frame, coords_data_crop_l[0],coords_data_crop_l[1],coords_data_crop_l[2], coords_data_crop_l[3])
        img_right_eye = utils.crop_image(frame, coords_data_crop_r[0],coords_data_crop_r[1],coords_data_crop_r[2], coords_data_crop_r[3])

        return img_left_eye, img_right_eye


    def get_eyes_center(self, points,frame_cropped_w, frame_cropped_h):
        points = points['95']

        #print("points:", points[0][0])
        for point in points:
            xl,yl = point[0][0] * frame_cropped_w, point[1][0] * frame_cropped_h
            xr,yr = point[2][0] * frame_cropped_w, point[3][0] * frame_cropped_h
            xn,yn = point[4][0] * frame_cropped_w, point[5][0] * frame_cropped_h


            left_eye_center = (xl,yl)
            right_eye_center = (xr,yr)


        return left_eye_center, right_eye_center