import cv2
import numpy as np
from model import Model
import math as m


class gazeEstimation():

    def __init__(self, MODEL_PATH, DEVICE):
        #super().__init__()
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
    
        
    def set_params(self, frame, left_eyes_frame, right_eyes_frame, head_pose_angle, left_eye_center, right_eye_center, initial_w, initial_h):
        
        head_pose_angle = np.array((head_pose_angle['angle_p_fc'][0][0], head_pose_angle['angle_r_fc'][0][0], head_pose_angle['angle_y_fc'][0][0]))
        #print(head_pose_angle)
        self.frame = frame

        self.left_eye_center = left_eye_center
        self.right_eye_center = right_eye_center

        self.left_eyes_frame = left_eyes_frame
        self.right_eyes_frame = right_eyes_frame
        self.head_pose_angle = head_pose_angle

        self.initial_w = initial_w
        self.initial_h = initial_h

    def get_inference_outputs(self):
        focal_length = 950.0
        scale = 100

        inputs = self.input_blobs()
        #print(self.left_eyes_frame.shape)
        prepro_img_left_eyes = self.preprocess_frame(self.left_eyes_frame)
        prepro_img_right_eyes = self.preprocess_frame(self.right_eyes_frame)
        inputs_model = {inputs[0]:self.head_pose_angle, inputs[1]:prepro_img_left_eyes, inputs[2]:prepro_img_right_eyes}
        gaze_vectors = self.inference(inputs_model)
        
        center_face = (self.frame.shape[0]/2, self.frame.shape[1]/2)

        self.frame = self.draw_axes(self.frame, center_face, self.head_pose_angle[2], self.head_pose_angle[0], self.head_pose_angle[1], scale, focal_length)
        frame =  self.draw_gaze_direction(self.frame, gaze_vectors)
        
        return frame, gaze_vectors['gaze_vector'][0] 
    
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

    def draw_axes(self,frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        
        cx = int(center_of_face[0] - 80)
        cy = int(center_of_face[1]+ 100)
        
        Rx = np.array([[1, 0, 0],
                    [0, m.cos(pitch), -m.sin(pitch)],
                    [0, m.sin(pitch), m.cos(pitch)]])
        
        Ry = np.array([[m.cos(yaw), 0, -m.sin(yaw)],
                    [0, 1, 0],
                    [m.sin(yaw), 0, m.cos(yaw)]])
        
        Rz = np.array([[m.cos(roll), -m.sin(roll), 0],
                    [m.sin(roll), m.cos(roll), 0],
                    [0, 0, 1]])
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        
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

    def build_camera_matrix(self, center_of_face, focal_length): 
    
        matrix = np.array(
                         [[focal_length, 0, center_of_face[0]],
                         [0, focal_length, center_of_face[1]],
                         [0, 0, 1]], dtype = "double"
                         )

        return matrix
    
    
    def draw_gaze_direction(self, frame, gaze_vector):

        x, y = gaze_vector["gaze_vector"][0, :2]
        frame = cv2.line(frame, (int(self.left_eye_center[0]),int(self.left_eye_center[1]+ 5)), (int(self.left_eye_center[0]+x*200), int(self.left_eye_center[1]-y*200)), (238, 130, 238), 2)
        frame = cv2.line(frame, (int(self.right_eye_center[0]),int(self.right_eye_center[1] + 5)), (int(self.right_eye_center[0]+x*200), int(self.right_eye_center[1]-y*200)), (238, 130, 238), 2)
        return  frame        
