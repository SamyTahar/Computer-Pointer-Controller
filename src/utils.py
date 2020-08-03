
import numpy as np
import cv2

def check_frame_shape_error(frame):
    print("shape before reshape: ", frame.shape)
    if frame.shape[0] == 0:
        print("[info]: warning frame.shape[0] equal O changed to 1 to avoid error ") 
        frame.resize((1,frame.shape[1]))
        print("shape after reshape case frame.shape[1]: ", frame.shape)
    
    if frame.shape[1] == 0:
        print("[info]: warning frame.shape[1] equal O changed to 1 to avoid error ")
        frame.resize((frame.shape[0], 1)) 
        print("shape after reshape case frame.shape[1]: ", frame.shape)

    return frame

def preprocess_frame(frame, image_input_shape):

    print("frame_shape: ",frame.shape, "self.image_input_shape[0][3]: ", image_input_shape[0][3], "self.image_input_shape[0][2]: ", image_input_shape[0][2])
    
    resize_frame = cv2.resize(frame, (image_input_shape[0][3], image_input_shape[0][2]), interpolation=cv2.INTER_AREA)
    resize_frame = resize_frame.transpose((2,0,1))
    resize_frame = resize_frame.reshape(1, *resize_frame.shape)

    return resize_frame


def crop_image(frame, y1,y2,x1,x2):
        
    return frame[y1:y2, x1:x2]

def flip_image_vertical(frame):
    return cv2.flip(frame, 1)

def flip_image_horizontal(frame):
    return cv2.flip(frame, 0)

def get_scale_factor(new_value, old_value):
    return new_value/old_value

def draw_visualisation(frame, face_data_points, eye_data_point, threshold, initial_w, initial_h):
    pass
