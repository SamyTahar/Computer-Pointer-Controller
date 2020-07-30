import line_profiler
profile=line_profiler.LineProfiler()
import atexit
atexit.register(profile.print_stats)
import numpy as np
import cv2
#from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def preprocess_frame(image_input_shape):
    image=cv2.imread('images/retail_image.jpg')
    resize_frame = cv2.resize(image, (image_input_shape[3], image_input_shape[2]))
    resize_frame = resize_frame.transpose((2,0,1))
    resize_frame = resize_frame.reshape(1, *resize_frame.shape)

    return resize_frame

def load_model(args):
    model=args.model
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    ie = IECore()
    net = ie.read_network(model=model_structure, weights=model_weights)
    
    input_name=next(iter(net.inputs))
    input_shape=net.inputs[input_name].shape

    exec_net = ie.load_network(network=net, device_name="CPU", num_requests=1)

    return exec_net, input_name, input_shape

@profile
def main(args):

    # Loading the Model
    exec_net, input_name, input_shape=load_model(args)

    # Reading and Preprocessing Image
    input_img=preprocess_frame(input_shape)

    exec_net.infer({input_name:input_img})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    
    args=parser.parse_args()
    main(args)