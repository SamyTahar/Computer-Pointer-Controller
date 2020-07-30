
from facedetection import FaceDetection
from headposeestimation import headPoseEstimation
from facelandmarks import FaceLandmarks
from gazeestimation import gazeEstimation
from input_feeder import InputFeeder
import cv2
import numpy as np
import math

#IMG = "../bin/detect_people.jpg"
IMG = "images/sam.jpg"
#IMG = "images/retail_image.jpg"
MODEL_FACE_DETECTION = "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001"
MODEL_GAZE_ESTIMATION = "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"
MODEL_HEAD_POSE_ESTIMATION = "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
MODEL_LANDMARKS = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"


def main():

    face_Detect = FaceDetection(MODEL_FACE_DETECTION)
    land_Marks = FaceLandmarks(MODEL_LANDMARKS)
    head_PoseEstimat = headPoseEstimation(MODEL_HEAD_POSE_ESTIMATION)
    gaze_Estimation = gazeEstimation(MODEL_GAZE_ESTIMATION)
    
    cap = cv2.VideoCapture(IMG)

    initial_w = int(cap.get(3))
    initial_h = int(cap.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4',fourcc, 10.0, (initial_w, initial_h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            
            face_Detect.set_params(frame, initial_w, initial_h)
            img_output, cropped_image, cropped_w, cropped_h = face_Detect.get_inference_outputs()            

            land_Marks.set_params(cropped_image, cropped_h, cropped_w)
            img_output, img_left_eye, img_right_eye= land_Marks.get_inference_outputs()
            
            head_PoseEstimat.set_params(cropped_image, cropped_w, cropped_h)
            output_image ,head_pose_angles = head_PoseEstimat.get_inference_outputs()
           
            gaze_Estimation.set_params(img_left_eye, img_right_eye, head_pose_angles, initial_w, initial_h)
            gaze_vectors_output = gaze_Estimation.get_inference_outputs()

            print(gaze_vectors_output)

            # write the flipped frame
            cv2.imwrite("output.jpg", output_image)
            #out.write(img_output)

            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
  



    """feed = InputFeeder(input_type='cam')
    feed.load_data()
    initial_h, initial_w = feed.get_input_size()
 
    for frame in feed.next_batch():
           
        prepro_img = faceDetect.preprocess_frame(frame)
        coords = faceDetect.inference(prepro_img)
        img_output = faceDetect.draw_outputs(coords, frame, 0.5 , initial_w, initial_h)

        # write the flipped frame
        feed.save_to_video(img_output)

    feed.close()
    """

if __name__ == '__main__':
    main()









#run.display_supported_layer()
#run.launch_cam()
