
from facedetection import FaceDetection
from headposeestimation import headPoseEstimation
from facelandmarks import FaceLandmarks
from gazeestimation import gazeEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController

import cv2
import numpy as np
import math
import utils
from argparse import ArgumentParser

#IMG = "../bin/detect_people.jpg"
IMG = "../bin/sam.jpg"
VIDEO = "../bin/demo.mp4"
#IMG = "images/retail_image.jpg"
MODEL_FACE_DETECTION = "models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001"
MODEL_LANDMARKS = "models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009"
MODEL_HEAD_POSE_ESTIMATION = "models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001"
MODEL_GAZE_ESTIMATION = "models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002"


def main(args):
    
    #init mouse controller class
    mouse_controller = MouseController('low','medium')

    #init model classes
    face_Detect = FaceDetection(args.face_detection, args.device)
    land_Marks = FaceLandmarks(args.landmarks, args.device)
    head_PoseEstimat = headPoseEstimation(args.head_pose_estimation, args.device)
    gaze_Estimation = gazeEstimation(args.gaze_estimation, args.device)

    #init input feeder class
    feed = InputFeeder(input_type=args.input_feed, input_file=args.path_feed)

    #load data input source from either image, video, cam according to the parameters passed by the user or the default one (video)
    cap = feed.load_data()

    #get the Height and width from the input source 
    initial_w, initial_h = feed.get_input_size()

    #Facedetection threshold prob from args
    THRESHOLD = args.prob_threshold
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4',fourcc, 10.0, (initial_w, initial_h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            #flip image
            frame = utils.flip_image_vertical(frame)

            #copy unmodifed frame
            original_frame = np.copy(frame)
            
            #Set facedetecetion parameters 
            face_Detect.set_params(frame,THRESHOLD, initial_w, initial_h)

            #Run facedetection inference
            confidence, data_face_detection_points = face_Detect.get_inference_outputs()
            
            if confidence >= THRESHOLD:
                
                cropped_frame, cropped_h, cropped_w = utils.crop_frame(
                                                    frame, 
                                                    data_face_detection_points[1],
                                                    data_face_detection_points[3],
                                                    data_face_detection_points[0],
                                                    data_face_detection_points[2])
                
                land_Marks.set_params(cropped_frame, cropped_h, cropped_w)
                
                left_eye_center_points ,right_eye_center_points, data_l_eye, data_r_eye, data_points_marks = land_Marks.get_inference_outputs()

                xomin = data_face_detection_points[0]
                yomin = data_face_detection_points[1]

                img_left_eye, _, _ = utils.crop_frame(
                                                    frame, 
                                                    data_l_eye[1]+ yomin,
                                                    data_l_eye[3]+ yomin,
                                                    data_l_eye[0]+ xomin,
                                                    data_l_eye[2]+ xomin)

                img_right_eye, _, _ = utils.crop_frame(
                                                    frame, 
                                                    data_r_eye[1]+ yomin,
                                                    data_r_eye[3]+ yomin,
                                                    data_r_eye[0]+ xomin,
                                                    data_r_eye[2]+ xomin)

                
                head_PoseEstimat.set_params(cropped_frame, cropped_w, cropped_h)
                head_pose_angles = head_PoseEstimat.get_inference_outputs()
                
                gaze_Estimation.set_params(img_left_eye, img_right_eye, head_pose_angles)
                gaze_vector_output = gaze_Estimation.get_inference_outputs()

                print("gaze_vector_output",gaze_vector_output)
                
                ####
                #eyes_concat = np.concatenate((img_left_eye,img_right_eye), axis=0)
                #eyes_concat_resized = cv2.resize(eyes_concat,(cropped_frame.shape[1] -200 ,cropped_frame.shape[0]), interpolation=cv2.INTER_AREA)
                #eyes_crop_out = np.concatenate((cropped_frame, eyes_concat_resized), axis=1)
                #display_visual = True
               
                frame = utils.draw_visualisation(frame, 
                                                data_face_detection_points, 
                                                data_points_marks, 
                                                head_pose_angles, 
                                                data_l_eye, 
                                                data_r_eye,
                                                gaze_vector_output)
        
                #if args.display_visual == True:
                #    original_frame = cv2.resize(original_frame,(cropped_frame.shape[1] +400 ,cropped_frame.shape[0]), interpolation=cv2.INTER_AREA)
                #    img_output = np.concatenate((original_frame,cropped_frame), axis=1)
                #else: 
                #    img_output = original_frame    
                ######

                #width, height = mouse_controller.getScreenSize()
                #currentMouseX, currentMouseY = mouse_controller.getCurrentMousePosition()
                #mouse_controller.move(*gaze_vector_output[:2])    
            
                
                #cv2.imwrite("output.jpg", img_left_eye)
                #out.write(img_output)
        
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()


def build_argparser():
   
    parser = ArgumentParser()
    
    parser.add_argument("-m1", "--face_detection", required=False, type=str, default=MODEL_FACE_DETECTION,
                        help="Path to your Face Detection model with a trained model.")
    
    parser.add_argument("-m2", "--landmarks", required=False, type=str, default=MODEL_LANDMARKS,
                        help="Path to your Land Marks model with a trained model.")
    
    parser.add_argument("-m3", "--head_pose_estimation", required=False, type=str, default=MODEL_HEAD_POSE_ESTIMATION,
                        help="Path to your Head Pose Estimation model with a trained model.")

    parser.add_argument("-m4", "--gaze_estimation", required=False, type=str, default=MODEL_GAZE_ESTIMATION,
                        help="Path to your Gaze estimation model with a trained model.")

    parser.add_argument("-i", "--input_feed", required=False, type=str, default='video',
                        help="select your type of feed image, video file or use \"cam\" keyword to use your webcam ")
    
    parser.add_argument("-pf", "--path_feed", required=False, type=str, default=VIDEO,
                        help="select your image path if you have set your input to image or video path if you have set your input to video")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.8,
                        help="Probability threshold for detections filtering"
                        "(0.8 by default)")
    
    parser.add_argument("-dis", "--display_visual", type=bool, default=False,
                        help="Display marks and head position for debug purpose | Value True display on False display off (bool type keep capitals)")

    return parser

if __name__ == '__main__':
    # Grab command line args

    args = build_argparser().parse_args()
    main(args)
