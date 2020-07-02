
"""
YOLO Network Descriptor

@author: Abdullahi S. Adamu
"""
from absl import logging
from darknet_config_generator.yolo_optimizers import *
from darknet_config_generator.yolo_preprocess import *
from darknet_config_generator.yolo_layers import Layer
from darknet_config_generator.yolo_network import get_yolov3
from darknet_config_generator.common import NL

class YOLONetwork:
    """
    YOLO Object Detection Network
    
    This is a network descriptor that is able to generate a darknet network configuration file.
    """
    def __init__(self, input_dim=(608,608,3), image_augmentation=YOLOImageAugmentation(), optimizer=YOLOOptimizer(), layers:Layer=[]):
        self.__HEADER__= '[net]'
        self.input_dim = input_dim
        self.img_aug = image_augmentation
        self.optimizer = optimizer
        self.layers = layers 
        
    def export(self, file_obj):
        """exports the given layer"""
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'# Network Dimensions{NL}')
        file_obj.write(f'width={self.input_dim[0]}{NL}')
        file_obj.write(f'height={self.input_dim[1]}{NL}')
        file_obj.write(f'channels={self.input_dim[2]}{NL}')
        file_obj.write(NL)
        
    def generate_config(self, save_to='net.cfg'):
        """ generates network configuration"""
        file_obj = open(save_to,'w')
        
        # export network
        self.export(file_obj)
        
        # optimizer
        if self.optimizer:
            self.optimizer.export(file_obj)
        
        # image augmentation
        if self.img_aug:
            self.img_aug.export(file_obj)
        
        # export layers
        if self.layers:
            for layer_i in self.layers:
                layer_i.export(file_obj)
    

def test():
    logging.set_verbosity('debug')
    img_aug = YOLOImageAugmentation()
    yolo_optimizer = YOLOOptimizer(batch_size=1, subdivisions=1, learning_rate=0.05, lr_decay_schedule={5000: 0.001, 2000:0.001})
    yolo_net = YOLONetwork(input_dim=(608,608,3),image_augmentation=img_aug, optimizer=yolo_optimizer, layers=get_yolov3())
    yolo_net.generate_config('./darknet_yolov3/generated.cfg')

if __name__ == '__main__':
    test()