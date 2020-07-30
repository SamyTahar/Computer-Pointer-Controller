'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IECore
import logging as log
import numpy as np

class Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device_name='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables. model_name, device='CPU', extensions=None
        '''
        self.ie = IECore()
        self.extension = extensions

        self.model_structure=model_name+'.xml'
        self.model_weights=model_name+'.bin'      
        
        self.device = device_name
        self.net = None
        self.layers_map = None
        self.exec_net = None

        try:
            self.net = self.ie.read_network(model=self.model_structure, weights=self.model_weights)
            self.layers_map = self.ie.query_network(network=self.net, device_name=device_name)
       
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?", e)
                
        self.input_blob = list(self.net.inputs.keys()) 
        self.output_blob = list(self.net.outputs.keys())
        
        self.input_shape = None #self.net.inputs[self.input_blob].shape
        self.output_shape = None #self.net.outputs[self.output_blob].shape

    
    def load_model(self):
        return self.net

    def get_input_blob(self):

        print(type(next(iter(self.net.inputs))), next(iter(self.net.inputs)) )
        return self.input_blob 

    def get_output_blob(self):
        return self.output_blob

    def get_input_shape(self):
        model_inputs_name = np.array(self.input_blob)
        array_data_inputs = [] 
        for i in range(len(model_inputs_name)): 
            array_data_inputs.append(self.net.inputs[model_inputs_name[i]].shape)        
        
        return np.array(array_data_inputs) 
    
    def get_output_shape(self):

        model_outputs_name = np.array(self.output_blob)
        array_data_outputs = []

        for i in range(len(model_outputs_name)): 
            array_data_outputs.append(self.net.outputs[model_outputs_name[i]].shape)
        
        return np.array(array_data_outputs)
     

    def load_network(self):
        return self.ie.load_network(self.net, self.device, num_requests=0)

    def get_inference(self, input_data):
        self.exec_net = self.load_network()

        return self.exec_net.infer(inputs=input_data)

    def get_infer_output(self, input_data):
        
        self.get_inference(input_data)
               
        return self.exec_net.requests[0].outputs  

    def get_supported_layer(self):
        for pair in self.layers_map.items():
            print("supported_layers: ", pair)

    def get_unsupported_layer(self):

        not_supported_layers = [l for l in self.net.layers.keys() if l not in self.layers_map]
        
        if len(not_supported_layers) != 0: 
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(self.device, ', '.join(not_supported_layers)))
        else:
            log.info("All layers are supported!")

    def get_model_name(self):
        return self.model_structure

    def get_openvino_version(self, device_name):
        return self.ie.get_versions(device_name)
    
"""     def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError """
