
"""
Image Augmentation 

@author: Abdullahi S. Adamu
"""

from darknet_config_generator.common import *


"""Image Augmentation"""
class YOLOImageAugmentation:
    def __init__(self, hue:float=0.1, saturation:float=1.5, exposure:float=1.5, angle:int=0):
        self.__HEADER__ = '# Image Augementation Parameters'
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.angle = angle
    def export(self, file_obj):
        """exports image augementation parameters to the file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'hue={self.hue}{NL}')
        file_obj.write(f'saturation={self.saturation}{NL}')
        file_obj.write(f'exposure={self.exposure}{NL}')
        file_obj.write(f'angle={self.angle}{NL}')
        file_obj.write(NL)