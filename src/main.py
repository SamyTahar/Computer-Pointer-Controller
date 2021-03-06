
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

import logging as log
log.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log.DEBUG)
#logging.basicConfig(filename='logs/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

VIDEO = "../bin/demo.mp4"

MODEL_FACE_DETECTION = "../bin/models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001"
MODEL_LANDMARKS = "../bin/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"
MODEL_HEAD_POSE_ESTIMATION = "../bin/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
MODEL_GAZE_ESTIMATION = "../bin/models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"


def main(args):
    
    #init mouse controller class
    mouse_controller = MouseController('low','medium')

    log.debug('Init model classes')
    #init model classes
    face_Detect = FaceDetection(args.face_detection, args.device)
    land_Marks = FaceLandmarks(args.landmarks, args.device)
    head_PoseEstimat = headPoseEstimation(args.head_pose_estimation, args.device)
    gaze_Estimation = gazeEstimation(args.gaze_estimation, args.device)

    #init input feeder class
    feed = InputFeeder(input_type=args.input_feed, input_file=args.path_feed)

    log.info('load input source ...')
    #load data input source from either image, video, cam according to the parameters passed by the user or the default one (video)
    cap = feed.load_data()

    #load video save parameters
    feed.load_video_save_params(name_export_video='output_video.mp4')

    #get the Height and width from the input source 
    initial_w, initial_h = feed.get_input_size()

    #Facedetection threshold prob from args
    THRESHOLD = args.prob_threshold
    
    log.info('Run models inferences ...')
    while(cap.isOpened()):

        for ret, frame in feed.next_batch():
        
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
                    #Crop main frame with face detection coordinates use to draw the rectangle
                    cropped_frame, cropped_h, cropped_w = utils.crop_frame(
                                                        frame, 
                                                        data_face_detection_points[1],
                                                        data_face_detection_points[3],
                                                        data_face_detection_points[0],
                                                        data_face_detection_points[2])
                    
                    land_Marks.set_params(cropped_frame, cropped_h, cropped_w)
                    
                    left_eye_center_points ,right_eye_center_points, data_l_eye, data_r_eye, data_points_marks = land_Marks.get_inference_outputs()

                    #Use the x,y points from face detection to display visualisation at the right position
                    xomin = data_face_detection_points[0]
                    yomin = data_face_detection_points[1]

                    #Crop left eye from data generated by landmarks detection model
                    img_left_eye, _, _ = utils.crop_frame(
                                                        frame, 
                                                        data_l_eye[1]+ yomin,
                                                        data_l_eye[3]+ yomin,
                                                        data_l_eye[0]+ xomin,
                                                        data_l_eye[2]+ xomin)

                    #Crop right eye from data generated by landmarks detection model
                    img_right_eye, _, _ = utils.crop_frame(
                                                        frame, 
                                                        data_r_eye[1]+ yomin,
                                                        data_r_eye[3]+ yomin,
                                                        data_r_eye[0]+ xomin,
                                                        data_r_eye[2]+ xomin)

                    #Head pose estmisation model face detection copped_frame output (roll, pitch, yaw)
                    head_PoseEstimat.set_params(cropped_frame, cropped_w, cropped_h)
                    head_pose_angles = head_PoseEstimat.get_inference_outputs()
                    
                    #Gaze estimation model output vector for eyes direction
                    gaze_Estimation.set_params(img_left_eye, img_right_eye, head_pose_angles)
                    gaze_vector_output = gaze_Estimation.get_inference_outputs()
                    
                    ####
                    #eyes_concat = np.concatenate((img_left_eye,img_right_eye), axis=0)
                    #eyes_concat_resized = cv2.resize(eyes_concat,(cropped_frame.shape[1] -200 ,cropped_frame.shape[0]), interpolation=cv2.INTER_AREA)
                    #eyes_crop_out = np.concatenate((cropped_frame, eyes_concat_resized), axis=1)
                    #display_visual = True
                

                    #Display visualisation according to user cli arguments 
                    if args.display_visual == "True":
                        #original_frame = cv2.resize(original_frame,(cropped_frame.shape[1] +400 ,cropped_frame.shape[0]), interpolation=cv2.INTER_AREA)
                        #img_output = np.concatenate((original_frame,cropped_frame), axis=1)
                        frame = utils.draw_visualisation(frame, 
                                                    data_face_detection_points, 
                                                    data_points_marks, 
                                                    head_pose_angles, 
                                                    data_l_eye, 
                                                    data_r_eye,
                                                    gaze_vector_output)
                    else: 
                        frame = original_frame    

                    #show the frame(s) in realtime
                    cv2.imshow('frame',frame)
                    
                    #if the user chose "image" as input will be saved 
                    if args.input_feed == 'image':
                        cv2.imwrite("../bin/output.jpg", frame)

                    #if the user chose "video" as input will be saved 
                    if args.input_feed =='video' or 'cam':
                        #save the feed to video
                        feed.save_to_video(frame)

                    if args.mouse_move == "True":
                        pass
                        mouse_controller.move(*gaze_vector_output[:2])      
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        # Release everything if job is finished   
        feed.close()
    log.info('End inferences ...')

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
                        help="select your type of feed \"image\", \"video\" file or use \"cam\" keyword to use your webcam ")
    
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
    
    parser.add_argument("-dis", "--display_visual", type=str, default="False",
                        help="Display marks and head position for debug purpose | Value True display on False display off (bool type keep capitals)")

    parser.add_argument("-mmove", "--mouse_move", type=str, default="False",
                        help="Activate the mouse move control by eyes mouvement")
    

    return parser

if __name__ == '__main__':
    # Grab command line args

    args = build_argparser().parse_args()

    log.info(f"args {args}")

    main(args)
