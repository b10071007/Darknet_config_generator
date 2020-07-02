"""
Connection Descriptors

@author: Abdullahi S. Adamu
"""
from darknet_config_generator.common import *

""" Connections """
class Connection:
    def __init__(self):
        pass
    
class RouteConnection(Connection):
    """Routing Connection"""
    def __init__(self, layers=[-4]):
        self.__HEADER__= '[route]'
        self.layers = layers
    def export(self, file_obj):
        """ exports the route connection to the file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        if len(self.layers) > 1:
            file_obj.write(f'layers={list_to_str(self.layers, space=True)}{NL}')
        else:
            file_obj.write(f'layers={self.layers[0]}{NL}')
        file_obj.write(NL)
        
class SkipConnection(Connection):
    """ Skip connection"""
    def __init__(self, from_layer=-3,activation=Activations.LINEAR):
        self.__HEADER__ = '[shortcut]'
        self.from_layer = from_layer
        self.activation = activation
    def export(self, file_obj):
        """ exports the skip connection to the file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'from={self.from_layer}{NL}')
        file_obj.write(f'activation={self.activation}{NL}')
        file_obj.write(NL) 