# Computer Pointer Controller

This app uses 4 different models to perform your mouse cursor move according to your eye movement you can either load a video or use your webcam.

The first model will detect your face then your face will be analyzed to extract your eyes and head position then a vector direction makes your mouse move.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Screen")

## Project Set Up and Installation

### Folder structure

```Bash
.
├── README.md
├── bin
│   ├── demo.mp4
│   ├── images
│   │   ├── inference.png
│   │   └── time_to_load_model.png
│   └── models
│       └── intel
│           ├── face-detection-adas-0001
│           │   ├── FP16
│           │   │   ├── face-detection-adas-0001.bin
│           │   │   └── face-detection-adas-0001.xml
│           │   └── FP32
│           │       ├── face-detection-adas-0001.bin
│           │       └── face-detection-adas-0001.xml
│           ├── face-detection-adas-binary-0001
│           │   └── FP32-INT1
│           │       ├── face-detection-adas-binary-0001.bin
│           │       └── face-detection-adas-binary-0001.xml
│           ├── gaze-estimation-adas-0002
│           │   ├── FP16
│           │   │   ├── gaze-estimation-adas-0002.bin
│           │   │   └── gaze-estimation-adas-0002.xml
│           │   └── FP32
│           │       ├── gaze-estimation-adas-0002.bin
│           │       └── gaze-estimation-adas-0002.xml
│           ├── head-pose-estimation-adas-0001
│           │   ├── FP16
│           │   │   ├── head-pose-estimation-adas-0001.bin
│           │   │   └── head-pose-estimation-adas-0001.xml
│           │   └── FP32
│           │       ├── head-pose-estimation-adas-0001.bin
│           │       └── head-pose-estimation-adas-0001.xml
│           └── landmarks-regression-retail-0009
│               ├── FP16
│               │   ├── landmarks-regression-retail-0009.bin
│               │   └── landmarks-regression-retail-0009.xml
│               └── FP32
│                   ├── landmarks-regression-retail-0009.bin
│                   └── landmarks-regression-retail-0009.xml
├── requirements.txt
└── src
    ├── facedetection.py
    ├── facelandmarks.py
    ├── gazeestimation.py
    ├── headposeestimation.py
    ├── input_feeder.py
    ├── logs
    │   └── app.log
    ├── main.py
    ├── metrics.py
    ├── model.py
    ├── mouse_controller.py
    ├── output_video.mp4
    └── utils.py
```

### Clone the repos
Note that the models are within the repos and are specific with the openvino toolkit version 2020.4

### Install Openvino Toolkit

Download and Install Openvino Toolkit follow the instruction from the [intel website](https://docs.openvinotoolkit.org/)

### Install Anaconda to manage your python env

Download and Install anaconda [here is the link](https://www.anaconda.com/products/individual)

Create your env on the CLI

```bash
conda create --name <name_env> python=3.7
```

Install python dependecies the requirements file is located inside the src folder 

```bash
pip install -r requirements.txt
```

Note that you can use the virtual env of your choice.

### Export env variable
To avoid typing a long path to the openvino toolkit you can export the path here an example: 

```bash
export OPENVINO_ZOO_TOOL="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/"
```

### download model within the model folder

Models are already on the repos however you can download them again via the intel model zoo in case of a new version of openvino.

```bash
python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name face-detection-adas-binary-0001

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name face-detection-adas-0001 --precisions FP32,FP16,INT8

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP32,FP16,INT8

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP32,FP16,INT8

python $OPENVINO_ZOO_TOOL/downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP32,FP16,INT8

```

## Demo

To run the app with your webcam use the following command

```bash
python main.py -i cam -dis True -mmove True
```

To run the app with the default video use the following command:

```bash
python main.py -i cam -dis True -mmove True
```

If you want to use you own video use the following command: 

```bash
python main.py -i "path/to/your/video" -dis True -mmove True
```

If you want only to display the model output without the mouse move function use the following command:
```bash
python main.py -i "path/to/your/video" -dis True
```
or with the default video: 

python main.py -i "path/to/your/video" -dis True  

*Move your mouse cursor to the corner of your screen to stop the app. You can deactivate either the model visualization of the mouse automatic moves please refer to the Documentation section*

## Documentation

Here is the command line help that you can see by typing:

```bash
python main.py --help
```

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
                        select your type of feed "image", "video" file or use
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
                        Value True display on False display off (bool type
                        keep capitals)
```

## CLI command package app

python deployment_manager.py --targets cpu --user_data /home/workspace --output_dir /home/workspace/vtune_project --archive_name test_name

## Benchmarks

Here some benchmarks table that shows the performance model loading and inference time per frame/s on CPU intel E7 only

|Model name | FP | Model loading time  |  Inference Time |
|--- |---|---|---|
|   face-detection-adas-0001| FP16 | 0.098 sec | 0.23 sec per request|
|   landmarks-regression-retail-0009| FP16 | 0.026 sec | 0.047 sec per request |
|   head-pose-estimation-adas-0001| FP16 | 0.027 sec | 0.069 sec per request |
|   gaze-estimation-adas-0002| FP16 | 0.031 sec | 0.079 sec per request |

|Model name | FP | Model loading time  |  Inference Time |
|--- |---|---|---|
|   face-detection-adas-0001| FP32 | 0.13 sec| 0.29 sec ec per request|
|   landmarks-regression-retail-0009| FP32 | 0.032 sec | 0.042 sec per request |
|   head-pose-estimation-adas-0001| FP32 | 0.049 sec| 0.066 sec per request |
|   gaze-estimation-adas-0002| FP32 | 0.057 sec | 0.068 sec per request |

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Loading time")

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Inference Time")

## Results

The loading model process is faster when the FP is 16 compared to the FP 32 obviously because reducing operation from FP32 to FP16 will also reduce the size of the file.

However, regarding the inference time between FP32 and FP16 is not that obvious I was expecting to see that the inference time from the FP16 model will be faster but it is only the case on the faceDetection model.

Regarding the accuracy, it as a good chance that it will be less accurate when using an FP16 model or an INT8. The accuracy can be measure only with a test dataset OpenVino workbench can generate a dataset but won't be able to give the accuracy of the model without a real dataset so if you are not the model owner it will be difficult to find the right dataset at the right format.

According to the data here we could mix the use of FP16 and FP32 as the loading time is not an issue we could use faceDectection FP16 model and then for others model we could use THE FP32. 

### Mac troubleshooting

*If the the frame is not display on a windows when you are calling cv2.imshow and you get this error :

```bash
"You might be loading two sets of Qt binaries into the same process. Check that all plugins are compiled against the right Qt binaries. Export DYLD_PRINT_LIBRARIES=1 and check that only one set of binaries are being loaded.
qt.qpa.plugin: Could not load the Qt platform plugin "cocoa" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: cocoa, minimal, offscreen."
```

try to install : pip install opencv-python-headless in your python env

*If your mouse is not moving check the Security Preferences > Security & Privacy > Privacy > Accessibility and grant your terminal.

Nb: Accessibility is the panel on the left hand side on the Privacy tab.
