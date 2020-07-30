import cv2
import numpy as np
from model import Model


class FaceDetection():

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
        self.coords = None

        #print("input_names:", inputs,  "inputs_shape: ",inputs_shape)
        #print("output_names:", outputs, "outputs_shape: ",outputs_shape)
    
        
    def set_params(self, frame, initial_w, initial_h):
        self.frame = frame
        self.initial_w = initial_w
        self.initial_h = initial_h

    def get_inference_outputs(self):

        inputs_facedetect = self.input_blobs()
        prepro_img_face = self.preprocess_frame(self.frame)
        inputs_face = {inputs_facedetect[0]:prepro_img_face}
        coords = self.inference(inputs_face)
        img_output, cropped_image = self.draw_boxes(coords, self.frame, 0.5 , self.initial_w, self.initial_h)
        
        cropped_h, cropped_w = cropped_image.shape[:2]
        
        return img_output, cropped_image, cropped_w, cropped_h
    
    def input_blobs(self):
        return self.model_loaded.get_input_blob()

    def output_blobs(self):
        return self.model_loaded.get_output_blob()    
    
    def preprocess_frame(self, frame):
        
        image_input_shape = self.model_loaded.get_input_shape()

        resize_frame = cv2.resize(frame, (image_input_shape[0][3], image_input_shape[0][2]), interpolation=cv2.INTER_AREA)
        resize_frame = resize_frame.transpose((2,0,1))
        resize_frame = resize_frame.reshape(1, *resize_frame.shape)

        return resize_frame

    def inference(self, input_data):
        
        return self.model_loaded.get_infer_output(input_data)

    def draw_boxes(self, coords, image, threshold , initial_w, initial_h):
        width = initial_w
        height = initial_h
        coords = coords['detection_out']
            
        for box in coords[0][0]: # Output shape is 1x1x200x7
            conf = box[2]

            if conf >= threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 10)

                croped_image = self.crop_image(image, ymin,ymax, xmin,xmax)

        return image, croped_image 
    
    
    def crop_image(self,img, y1,y2,x1,x2):
        
        return img[y1:y2, x1:x2]
         
