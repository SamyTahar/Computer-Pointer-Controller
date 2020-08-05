
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


def crop_frame(frame, y1,y2,x1,x2):
    
    cropped_frame = frame[y1:y2, x1:x2]
    cropped_h, cropped_w = cropped_frame.shape[:2]
    
    return cropped_frame, cropped_h, cropped_w

def flip_image_vertical(frame):
    return cv2.flip(frame, 1)

def flip_image_horizontal(frame):
    return cv2.flip(frame, 0)

def get_scale_factor(new_value, old_value):
    return new_value/old_value

def draw_visualisation(frame, data_face_detection_points, data_points_marks):
    # FaceDetection rectangle
    # TODO: Add landmark points
    # TODO: Add eye detection rectangles 
    # TODO: Add headpose estimation text function routine 
    # TODO: Add Gaze direction viz

    frame = facedetection_viz(frame, data_face_detection_points)
    frame = landmarks_points_viz(frame, data_face_detection_points, data_points_marks)
    
    return frame

def facedetection_viz(frame, data_face_detection):
    xmin, ymin, xmax, ymax = data_face_detection[0], data_face_detection[1], data_face_detection[2], data_face_detection[3]
    return cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 10)

def landmarks_points_viz(frame, data_face_detection_points, data_points_marks):
    xlmin, ymin = data_face_detection_points[0] , data_face_detection_points[1]

    xl,yl = data_points_marks[0], data_points_marks[1]
    xr,yr = data_points_marks[2], data_points_marks[3]
    xn,yn = data_points_marks[4], data_points_marks[5]

    cv2.circle(frame,(xl + xlmin ,yl + (ymin+5)), 5, (0,0,255), -1)
    cv2.circle(frame,(xr + xlmin ,yr + (ymin+5)), 5, (0,0,255), -1)
    cv2.circle(frame,(xn + xlmin ,yn + (ymin+5)), 20, (0,255,), -1)
    
    return frame
