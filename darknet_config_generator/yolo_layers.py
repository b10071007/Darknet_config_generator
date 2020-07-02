"""
Layer Descriptors

@author: Abdullahi S. Adamu
"""
from darknet_config_generator.common import NL, Activations

YOLO_ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]

""" Layers """
class Layer:
    """ Layer"""
    def __init__(self):
        pass
    def export(self):
        pass

class ConvolutionLayer(Layer):
    """ Convolution Layer"""
    def __init__(self, size:int=1, filters:int=255, stride:int=3, pad:int=1,
                         activation=Activations.LEAKY_RELU.value, batch_normalize:bool=True):
        self.__HEADER__='[convolutional]'
        self.size = size
        self.filters = filters
        self.stride = stride
        self.pad = pad
        self.activation = activation
        self.batch_normalize = batch_normalize
        
    def export(self, file_obj):
        """ writes to layer to file object"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        if self.batch_normalize:
            file_obj.write(f'batch_normalize={int(self.batch_normalize)}\n')
        file_obj.write(f'size={self.size}\n')
        file_obj.write(f'stride={self.stride}\n')
        file_obj.write(f'pad={self.pad}\n')
        file_obj.write(f'filters={self.filters}\n')
        file_obj.write(f'activation={self.activation}\n')
        file_obj.write(NL)


class SoftmaxLayer(Layer):
    """
    Softmax Layer
    """

    def __init__(self, groups=1):
        self.__HEADER__ = '[softmax]'
        self.groups = groups

    def export(self, file_obj):
        """exports to fileobject"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        file_obj.write(f'groups={self.groups}\n')
        file_obj.write(NL)


class MaxPoolingLayer(Layer):
    """
    MaxPooling layer
    """
    def __init__(self, size:int=3, stride:int=2, padding:int=0):
        self.__HEADER__ = '[maxpool]'
        self.size = size
        self.stride = stride
        self.padding = padding

    def export(self, file_obj):
        """exports maxpooling layer to file"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        file_obj.write(f'size={self.size}\n')
        file_obj.write(f'stride={self.stride}\n')
        file_obj.write(f'padding={self.padding}\n')
        file_obj.write(NL)

class FullyConnectedLayer(Layer):
    """
    Fully Connected Layer
    """
    def __init__(self, size:int=1000, activation=Activations.LINEAR.value):
        self.__HEADER__ = '[connected]'
        self.size = size
        self.activation = activation
    
    def export(self, file_obj):
        """exports to fileobject"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        file_obj.write(f'output={self.size}\n')
        file_obj.write(f'activation={self.activation}\n')
        file_obj.write(NL)

class DropOutLayer(Layer):
    """
    DropOut Layer
    """

    def __init__(self, dropout_prob:float=0.5):
        self.__HEADER__ = '[dropout]'
        self.dropout_prob = dropout_prob

    def export(self, file_obj):
        """exports dropout layer to the config file"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        file_obj.write(f'probability={self.dropout_prob}\n')
        file_obj.write(NL)

class YOLOLayer(Layer):
    """ 
    Object Detection Layer
    
    This is normally placed at different levels of the network to perform detections at different resolutions
    """
    def __init__(self, anchors:list=YOLO_ANCHORS, num_classes:int=80, jitter:float=0.5, masks:list=[6,7,8],
                        ignore_thresh:float=0.5, truth_thresh:float=1.0, random:bool=True):
        self.__HEADER__ = '[yolo]'
        self.masks = masks
        self.anchors = anchors
        self.num_anchors = len(anchors)//2
        self.classes = num_classes
        self.jitter = jitter
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.random = int(random)
        
    def export(self, file_obj):
        """exports the layer to the given file object"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        file_obj.write(f'mask={list_to_str(self.masks, space=False)}\n')
        file_obj.write(f'anchors={anchors_to_str(self.anchors)}\n')
        file_obj.write(f'classes={self.classes}\n')
        file_obj.write(f'num={self.num_anchors}\n')
        file_obj.write(f'jitter={self.jitter}\n')
        file_obj.write(f'ignore_thresh={self.ignore_thresh}\n')
        file_obj.write(f'truth_thresh={self.truth_thresh}\n')
        file_obj.write(f'random={self.random}\n')
        
class UpsampleLayer(Layer):
    """ Upsampling Layer"""
    def __init__(self, stride:int=2):
        self.__HEADER__ = '[upsample]'
        self.stride = stride
        
    def export(self, file_obj):
        """exports the layer to the given file object"""
        # file_obj.write(f'\n')
        file_obj.write(f'{self.__HEADER__}\n')
        file_obj.write(f'stride={self.stride}\n')
        file_obj.write(NL)

