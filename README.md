# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation

*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

### Export env variable

export OPENVINO_ZOO_TOOL="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/"

### download model

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name face-detection-adas-binary-0001

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name face-detection-adas-0001 --precisions FP32,FP16,INT8

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP32,FP16,INT8

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP32,FP16,INT8

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP32,FP16,INT8

## Demo

*TODO:* Explain how to run a basic demo of your model.

## CLI command package app

python deployment_manager.py --targets cpu --user_data /home/workspace --output_dir /home/workspace/vtune_project --archive_name test_name

## Documentation

*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m1 FACE_DETECTION, --face_detection FACE_DETECTION
                        Path to your Face Detection model with a trained
                        model.
  -m2 LANDMARKS, --landmarks LANDMARKS
                        Path to your Land Marks model with a trained model.
  -m3 HEAD_POSE_ESTIMATION, --head_pose_estimation HEAD_POSE_ESTIMATION
                        Path to your Head Pose Estimation model with a trained
                        model.
  -m4 GAZE_ESTIMATION, --gaze_estimation GAZE_ESTIMATION
                        Path to your Gaze estimation model with a trained
                        model.
  -i INPUT_FEED, --input_feed INPUT_FEED
                        select your type of feed image, video file or use
                        "cam" keyword to use your webcam
  -pf PATH_FEED, --path_feed PATH_FEED
                        select your image path if you have set your input to
                        image or video path if you have set your input to
                        video
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.8 by
                        default)
  -dis DISPLAY_VISUAL, --display_visual DISPLAY_VISUAL
                        Display marks and head position for debug purpose |
                        Value True display on False display off (bool type keep capitals)
```

## Benchmarks

*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results

*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions

This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference

If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases

There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

### Mac troubleshooting

- If the the frame is not display on a windows when you are calling cv2.imshow and you get this error :

"You might be loading two sets of Qt binaries into the same process. Check that all plugins are compiled against the right Qt binaries. Export DYLD_PRINT_LIBRARIES=1 and check that only one set of binaries are being loaded.
qt.qpa.plugin: Could not load the Qt platform plugin "cocoa" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: cocoa, minimal, offscreen."

try to install : pip install opencv-python-headless in your python env

- If your mouse is not moving check the Security Preferences > Security & Privacy > Privacy > Accessibility and grant your terminal.

Nb: Accessibility is the panel on the left hand side on the Privacy tab.
