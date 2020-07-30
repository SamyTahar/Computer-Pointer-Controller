import cv2
import numpy as np
from model import Model


class Model_Loader(Model):

    def __init__(self, MODEL_PATH):
        #super().__init__()
        self.model_loaded = Model(MODEL_PATH)
        self.model_loaded.get_unsupported_layer()
        model_name =  self.model_loaded.get_model_name()
        print("log[info]: Model input shape " + model_name +" ", self.model_loaded.get_input_shape())
        print("log[info]: Model output shape " + model_name +" ", self.model_loaded.get_output_shape())

    def input_blobs(self):
        return self.model_loaded.get_input_blob()

    #def input_shapes(self):
    #    return self.model_loaded.get_input_shape()

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
         


    def draw_point(self,img, points, initial_w, initial_h):
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
