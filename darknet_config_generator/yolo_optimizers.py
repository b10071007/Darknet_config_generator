
"""
Optimizer Descriptor

@author: Abdullahi S. Adamu
"""


from darknet_config_generator.common import *


""" Learning Rate Decay Policies """
class ScheduledLRDecay:
    """
    Learning rate decay policy
    """
    def __init__(self, lr_decay_schedule={400000:0.1, 450000:0.1}):
        self.__HEADER__= '# LR Policy'
        self.policy = LearningRateDecayPolicy.SCHEDULED
        self.lr_decay_schedule = lr_decay_schedule

    def export(self, file_obj):
        """ exports learning rate decay policy """
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'policy={self.policy}{NL}')
        file_obj.write(f'steps={list_to_str(self.lr_decay_schedule.keys())}{NL}')
        file_obj.write(f'scales={list_to_str(self.lr_decay_schedule.values())}{NL}')

"""Network Optimization """
class YOLOOptimizer:
    def __init__(self, learning_rate:float=0.001, batch_size=64, subdivisions=64, num_gpus:int=2,
                     policy=LearningRateDecayPolicy.SCHEDULED, momentum=0.9, lr_decay=0.0005,
                     lr_decay_schedule={400000:0.1, 450000:0.1}, burn_in:int=1000, batches_per_class=2000, num_classes=80):
        self.__HEADER__= '# Optimization Parameters'
        self.batch = batch_size
        self.subdivisions = subdivisions
        self.num_gpus = num_gpus
        self.burn_in = burn_in * self.num_gpus
        self.max_batches = batches_per_class * num_classes
        self.policy = policy
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.lr_decay_schedule = lr_decay_schedule
   
    def export(self, file_obj):
        """exports the optimizer to the given file object"""
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'batch={self.batch}{NL}')
        file_obj.write(f'subdivisions={self.subdivisions}{NL}')
        file_obj.write(f'decay={self.lr_decay}{NL}')
        file_obj.write(f'learning_rate={self.learning_rate}{NL}')
        file_obj.write(f'momentum={self.momentum}{NL}')
        file_obj.write(f'burn_in={self.burn_in}{NL}')
        file_obj.write(f'max_batches={self.max_batches}{NL}')
        file_obj.write(f'policy={self.policy}{NL}')
        file_obj.write(f'steps={list_to_str(self.lr_decay_schedule.keys())}{NL}')
        file_obj.write(f'scales={list_to_str(self.lr_decay_schedule.values())}{NL}')
        file_obj.write(NL)