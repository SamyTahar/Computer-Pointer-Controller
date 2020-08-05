
import numpy as np
import cv2
import math as m
from decimal import Decimal

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

def draw_visualisation(frame, data_face_detection_points, data_points_marks, head_pose_angles, data_l_eye, data_r_eye):
    # FaceDetection rectangle
    # landmark points
    # Eye detection rectangles 
    # headpose estimation text function routine 
    # TODO: Add Gaze direction viz

    frame = face_detection_viz(frame, data_face_detection_points)
    frame = landmarks_points_viz(frame, data_face_detection_points, data_points_marks)
    frame = eye_detection_viz(frame, data_face_detection_points, data_l_eye, data_r_eye)
    frame = head_pose_angle_text(frame, head_pose_angles)
    
    return frame

def face_detection_viz(frame, data_face_detection_points):
    xmin, ymin, xmax, ymax = data_face_detection_points[0], data_face_detection_points[1], data_face_detection_points[2], data_face_detection_points[3]
    return cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 10)

def landmarks_points_viz(frame, data_face_detection_points, data_points_marks):
    xomin = data_face_detection_points[0]
    yomin = data_face_detection_points[1]

    xl,yl = data_points_marks[0], data_points_marks[1]
    xr,yr = data_points_marks[2], data_points_marks[3]
    xn,yn = data_points_marks[4], data_points_marks[5]

    cv2.circle(frame,(xl + xomin ,yl + (yomin) + 5), 5, (0,0,255), -1)
    cv2.circle(frame,(xr + xomin ,yr + (yomin + 5)), 5, (0,0,255), -1)
    cv2.circle(frame,(xn + xomin ,yn + (yomin + 5)), 20, (0,255,), -1)
    
    return frame

def eye_detection_viz(frame, data_face_detection_points, data_l_eye, data_r_eye):
    
    xomin = data_face_detection_points[0]
    yomin = data_face_detection_points[1]

    xlmin, ylmin, xlmax, ylmax = data_l_eye[0], data_l_eye[1], data_l_eye[2], data_l_eye[3]
    xrmin, yrmin, xrmax, yrmax = data_r_eye[0], data_r_eye[1], data_r_eye[2], data_r_eye[3]

    print( "xlmin, ylmin, xlmax, ylmax", xlmin, ylmin, xlmax, ylmax)
    cv2.rectangle(frame, (xlmin + xomin, ylmin + yomin), (xlmax + xomin , ylmax + yomin), (0, 55, 255), 2)	
    cv2.rectangle(frame, (xrmin + xomin, yrmin + yomin), (xrmax + xomin, yrmax + yomin), (0, 55, 255), 2)

    return frame

def head_pose_angle_text(frame, head_pose_angles):

    yaw = head_pose_angles['angle_y_fc'][0][0]
    pitch = head_pose_angles['angle_p_fc'][0][0]
    roll = head_pose_angles['angle_p_fc'][0][0]

    roll = Decimal(str(roll))
    pitch = Decimal(str(pitch))
    yaw = Decimal(str(yaw))

    text_1 =  f"Roll  : {round(roll,2)} degrees"
    text_2 =  f"pitch : {round(pitch,2)} degrees"  
    text_3 =  f"yaw  : {round(yaw,2)} degrees"    
   
    position_x = 100

    cv2.putText(frame, text_1, (position_x,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(frame, text_2 , (position_x,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, text_3 , (position_x,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)   

    return frame
