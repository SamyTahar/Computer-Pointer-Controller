'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type ,input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file

        self.initial_w = None
        self.initial_h = None
        self.out = None

            
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)
        
        self.initial_w = int(self.cap.get(3))
        self.initial_h = int(self.cap.get(4))

        return self.cap

    def get_input_size(self):
        return  self.initial_w, self.initial_h

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for _ in range(50):
                ret, frame=self.cap.read()
            yield ret ,frame

    def load_video_save_params(self, name_export_video='output_video.mp4' ):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter(name_export_video ,fourcc, 10.0, (self.initial_w, self.initial_h))

    def save_to_video(self, frame):
        self.out.write(frame) 

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
            self.out.release()