
import numpy as np
import cv2
import math as m
from decimal import Decimal

def preprocess_frame(frame, image_input_shape):
    
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

def draw_visualisation(frame, data_face_detection_points, data_points_marks, head_pose_angles, data_l_eye, data_r_eye, gaze_vector_output):
    # FaceDetection rectangle
    # landmark points
    # Eye detection rectangles 
    # headpose estimation text function routine 
    # Draw axis
    # Gaze direction viz

    frame = face_detection_viz(frame, data_face_detection_points)
    #frame = landmarks_points_viz(frame, data_face_detection_points, data_points_marks)
    frame = eye_detection_viz(frame, data_face_detection_points, data_l_eye, data_r_eye)
    frame = head_pose_angle_text(frame, head_pose_angles)
    frame = draw_axes(frame, head_pose_angles, data_face_detection_points)
    frame = draw_gaze_direction(frame, gaze_vector_output, data_points_marks, data_face_detection_points)
    
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

    cv2.putText(frame, text_1, (position_x,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(frame, text_2 , (position_x,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, text_3 , (position_x,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)   

    return frame


def draw_axes(frame, head_pose_angles, data_face_detection_points):
     # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    xomin = data_face_detection_points[0]
    yomin = data_face_detection_points[1]
    xomax = data_face_detection_points[2]
    yomax = data_face_detection_points[3]

    center_of_face = ((xomin + xomax/8), (yomin + yomax/2))

    yaw = head_pose_angles['angle_y_fc'][0][0]
    pitch = head_pose_angles['angle_p_fc'][0][0]
    roll = head_pose_angles['angle_r_fc'][0][0] 

    focal_length = 950.0
    scale = 100

    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    
    cx = int(center_of_face[0])
    cy = int(center_of_face[1] + 100)
    
    Rx = np.array([[1, 0, 0],
                [0, m.cos(pitch), -m.sin(pitch)],
                [0, m.sin(pitch), m.cos(pitch)]])
    
    Ry = np.array([[m.cos(yaw), 0, -m.sin(yaw)],
                [0, 1, 0],
                [m.sin(yaw), 0, m.cos(yaw)]])
    
    Rz = np.array([[m.cos(roll), -m.sin(roll), 0],
                [m.sin(roll), m.cos(roll), 0],
                [0, 0, 1]])
 
    R = Rz @ Ry @ Rx

    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    
    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    
    return frame    

def build_camera_matrix(center_of_face, focal_length): 

    matrix = np.array(
                        [[focal_length, 0, center_of_face[0]],
                        [0, focal_length, center_of_face[1]],
                        [0, 0, 1]], dtype = "double"
                        )

    return matrix
    
    
def draw_gaze_direction(frame, gaze_vector, data_points_marks, data_face_detection_points):

    xomin = data_face_detection_points[0]
    yomin = data_face_detection_points[1]
    xomax = data_face_detection_points[2]
    yomax = data_face_detection_points[3]

    left_eye_center = data_points_marks[0], data_points_marks[1]
    right_eye_center = data_points_marks[2], data_points_marks[3]

    xl1 = int(left_eye_center[0] + xomin)
    yl1 = int(left_eye_center[1] + yomin)

    xr1 = int(right_eye_center[0] + xomin)
    yr1 = int(right_eye_center[1] + yomin)

    x, y = gaze_vector[:2]
    
    frame = cv2.line(frame, (xl1,yl1), (int(xl1+x*300), int(yl1-y*300)), (238, 130, 238), 2)
    frame = cv2.line(frame, (xr1,yr1), (int(xr1+x*300), int(yr1-y*300)), (238, 130, 238), 2)
    return  frame
