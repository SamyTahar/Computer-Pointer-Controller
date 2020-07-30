
import cv2
import numpy as np
from model import Model


class FaceLandmarks(Model):

    def __init__(self, MODEL_PATH):
        #super().__init__()
        self.model_loaded = Model(MODEL_PATH)
        self.model_loaded.get_unsupported_layer()
        #model_name =  self.model_loaded.get_model_name()
        #print("log[info]: Model input shape " + model_name +" ", self.model_loaded.get_input_shape())
        #print("log[info]: Model output shape " + model_name +" ", self.model_loaded.get_output_shape())

        self.frame = None
        self.initial_w = None
        self.initial_h = None

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
        prepro_img_face = self.preprocess_frame(self.frame)
        inputs_to_feed = {inputs_model[0]:prepro_img_face}
        
        points = self.inference(inputs_to_feed)

        img, img_left_eye, img_left_right = self.draw_box_eyes(self.frame ,points,self.initial_h, self.initial_w)
        
        return img, img_left_eye, img_left_right
    
    def preprocess_frame(self, frame):
        
        image_input_shape = self.model_loaded.get_input_shape()

        resize_frame = cv2.resize(frame, (image_input_shape[0][3], image_input_shape[0][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        return self.model_loaded.get_infer_output(input_data)
    
    def crop_image(self,img, y1,y2,x1,x2):
        
        return img[y1:y2, x1:x2]
         

    def draw_box_eyes(self, img, points, initial_w, initial_h):
        
        points = points['95']

        #print("points:", points[0][0])
        for point in points:
            xl,yl = point[0][0] * initial_w, point[1][0] * initial_h
            xr,yr = point[2][0] * initial_w, point[3][0] * initial_h
            xn,yn = point[4][0] * initial_w, point[5][0] * initial_h

            # make box for left eye 
            xlmin = xl-120
            ylmin = yl-120
            xlmax = xl+120
            ylmax = yl+120
            
            # make box for right eye 
            xrmin = xr-120
            yrmin = yr-120
            xrmax = xr+120
            yrmax = yr+120

            img_left_eye = self.crop_image(img, int(ylmin),int(ylmax),int(xlmin), int(xlmax))
            img_left_right = self.crop_image(img, int(yrmin),int(yrmax), int(xrmin), int(xrmax))

            #cv2.circle(img,(xl,yl), 10, (0,0,255), -1)
            #cv2.circle(img,(xr,yr), 10, (0,0,255), -1)
            #cv2.circle(img,(xn,yn), 10, (0,255,), -1)


           
            #cv2.rectangle(img, (xlmin, ylmin), (xlmax, ylmax), (0, 55, 255), 10)
            #cv2.rectangle(img, (xrmin, yrmin), (xrmax, yrmax), (0, 55, 255), 10)

            
        
        return img, img_left_eye, img_left_right