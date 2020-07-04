from darknet_config_generator.yolo_connections import *
from darknet_config_generator.yolo_layers import *
from darknet_config_generator.yolo_optimizers import *
from darknet_config_generator.yolo_preprocess import *
from darknet_config_generator.common import *
from darknet_config_generator.yolo_darknet import YOLONetwork

def _get_2x_conv_block(filters=64):
    return [ConvolutionLayer(filters=filters, size=3, stride=1),
            ConvolutionLayer(filters=filters, size=3, stride=1),
            MaxPoolingLayer(size=2, stride=2)]

def _get_3x_conv_block(filters=64, pool_size=2, pool_stride=2, pool_pad=0):
    return [ConvolutionLayer(filters=filters, size=3, stride=1),
            ConvolutionLayer(filters=filters, size=3, stride=1),
            ConvolutionLayer(filters=filters, size=3, stride=1),
            MaxPoolingLayer(size=pool_size, stride=pool_stride, padding=pool_pad)]

def _get_RFB_s_block(from_layer=0, in_filters=512, out_filters=512):
    
    inter_filters = in_filters // 4

    if from_layer == 0:
        branch_0 = [ConvolutionLayer(filters=inter_filters, size=1, stride=1),
                    ConvolutionLayer(filters=inter_filters, size=3, stride=1)] # 2
    else:
        branch_0 = [RouteConnection(layers=[from_layer]),
                    ConvolutionLayer(filters=inter_filters, size=1, stride=1),
                    ConvolutionLayer(filters=inter_filters, size=3, stride=1)] # 2

    branch_1 = [RouteConnection(layers=[-3 + from_layer]),
                ConvolutionLayer(filters=inter_filters, size=1, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1)] # 4
    
    branch_2 = [RouteConnection(layers=[-7 + from_layer]),
                ConvolutionLayer(filters=inter_filters, size=1, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1)] # 4
    
    branch_3 = [RouteConnection(layers=[-11 + from_layer]),
                ConvolutionLayer(filters=inter_filters, size=1, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1),
                ConvolutionLayer(filters=inter_filters, size=3, stride=1)] # 5
    
    concate = [RouteConnection(layers=[-1, -6, -10, -14]),
               ConvolutionLayer(filters=out_filters, size=1, stride=1)] # 2
    
    
    
    shortcut = [RouteConnection(layers=[-18 + from_layer]),
                ConvolutionLayer(filters=out_filters, size=1, stride=1),
                SkipConnection(from_layer=-3)] # 3
    
    return branch_0 + branch_1 + branch_2 + branch_3 + concate + shortcut

def _get_RFB_block(from_layer=0, in_filters=512, out_filters=512):
    
    inter_filters = in_filters // 8

    if from_layer == 0:
        branch_0 = [ConvolutionLayer(filters=2*inter_filters, size=1, stride=2),
                    ConvolutionLayer(filters=2*inter_filters, size=3, stride=1)] # 2
    else:
        branch_0 = [RouteConnection(layers=[from_layer]),
                    ConvolutionLayer(filters=2*inter_filters, size=1, stride=2),
                    ConvolutionLayer(filters=2*inter_filters, size=3, stride=1)] # 2

    branch_1 = [RouteConnection(layers=[-3 + from_layer]),
                ConvolutionLayer(filters=1*inter_filters, size=1, stride=1),
                ConvolutionLayer(filters=2*inter_filters, size=3, stride=2),
                ConvolutionLayer(filters=2*inter_filters, size=3, stride=1)] # 4
    
    branch_2 = [RouteConnection(layers=[-7 + from_layer]),
                ConvolutionLayer(filters=1*inter_filters, size=1, stride=1),
                ConvolutionLayer(filters=int(3/2*inter_filters), size=3, stride=1),
                ConvolutionLayer(filters=2*inter_filters, size=3, stride=2),
                ConvolutionLayer(filters=2*inter_filters, size=3, stride=1)] # 5
    
    concate = [RouteConnection(layers=[-1, -6, -10]),
               ConvolutionLayer(filters=out_filters, size=1, stride=1)] # 2
    
    shortcut = [RouteConnection(layers=[-14 + from_layer]),
                ConvolutionLayer(filters=out_filters, size=1, stride=2),
                SkipConnection(from_layer=-3)] # 3
    
    return branch_0 + branch_1 + branch_2 + concate + shortcut

def _get_vgg_net():
    layers = []
    layers += _get_2x_conv_block(filters=64)    # conv_1
    layers += _get_2x_conv_block(filters=128)   # conv_2
    layers += _get_3x_conv_block(filters=256)   # conv_3
    layers += _get_3x_conv_block(filters=512)   # conv_4
    layers += _get_3x_conv_block(filters=512, pool_size=3, pool_stride=1, pool_pad=1)   # conv_5
    layers += [ConvolutionLayer(filters=1024, size=3, stride=1, pad=1)] # conv_6
    layers += [ConvolutionLayer(filters=1024, size=1, stride=1, pad=1)] # conv_7
    return layers
    
def _get_head(from_layer=0, filters=512, yolo_mask=[0,1,2], num_classes=1):
    
    layers = []
    if from_layer == 0 or from_layer == -1:
        pass
    else:
        layers += [RouteConnection(layers=[from_layer])]

    layers += [ConvolutionLayer(filters=filters//2, size=1, stride=1, pad=1),
               ConvolutionLayer(filters=filters, size=3, stride=1, pad=1),
               ConvolutionLayer(filters=filters//2, size=1, stride=1, pad=1),
               ConvolutionLayer(filters=filters, size=3, stride=1, pad=1),
               ConvolutionLayer(filters=(num_classes+5)*3, size=3, stride=1, pad=1),
               YOLOLayer(num_classes=num_classes, masks=yolo_mask)]
    return layers

#----------------------------------------------------------------------------------------#

class Net_generator():
    def __init__(self, num_classes=1, augmentation = YOLOImageAugmentation(), optimizer = YOLOOptimizer(),
                 save_dir="./", input_size=512):
        self.layers = []
        self.num_layers = 0
        self.augmentation = augmentation
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.head_idx = []
    
    def add_layer(self, layer_list):
        num_added_layers = len(layer_list)
        # print( "Add layers from {} to {}.".format(self.num_layers+1, self.num_layers + num_added_layers) )
        print( "layer_{} -> layer_{}  ".format(self.num_layers+1, self.num_layers + num_added_layers))
        self.layers += layer_list
        self.num_layers += num_added_layers

    def reserve_head(self):
        self.head_idx.append(self.num_layers)

    def generate_config(self, fName):
        print("\nGenerate Network with {} layers.".format(self.num_layers))
        net = YOLONetwork(input_dim=(512,512,3), layers=self.layers, image_augmentation=augmentation, 
                          optimizer=optimizer)
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = os.path.join(self.save_dir, fName)
        print("Output file to \"{}\" \n".format(filepath))
        net.generate_config(filepath)

    def print_num_layers(self):
        print( "Number of layers = {}.".format(self.num_layers) )

#----------------------------------------------------------------------------------------#

def Generate_RFB_vgg_512(num_classes, augmentation, optimizer, save_dir):

    net = Net_generator(num_classes, augmentation, optimizer, save_dir, input_size=300)

    # Backbone
    print("\n[Backbone]")
    net.add_layer( _get_vgg_net() ) # 1~20
    
    # Neck
    print("\n[Neck]")
    net.add_layer( _get_RFB_s_block(from_layer=-8, in_filters=512, out_filters=512) ) # 21~41
    net.reserve_head()
    net.add_layer( _get_RFB_block(from_layer=-22, in_filters=1024, out_filters=1024) ) # 42~58
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=1024, out_filters=512) ) # 59~74
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=512, out_filters=256) ) # 75~90
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=512, out_filters=256) ) # 91~106
    net.reserve_head()
    print("\nReserve head on {}".format(net.head_idx))

    # Head
    print("\n[Head]")
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-1]) + 1), filters=256, yolo_mask=[0,1,2], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-2]) + 1), filters=256, yolo_mask=[3,4,5], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-3]) + 1), filters=512, yolo_mask=[6,7,8], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-4]) + 1), filters=1024, yolo_mask=[9,10,11], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-5]) + 1), filters=512, yolo_mask=[12,13,14], num_classes=num_classes) )

    net.generate_config("RFB_vgg_512.cfg")


def Generate_RFB_vgg_300(num_classes, augmentation, optimizer, save_dir):

    net = Net_generator(num_classes, augmentation, optimizer, save_dir, input_size=300)

    # Backbone
    print("\n[Backbone]")
    net.add_layer( _get_vgg_net() ) # 1~20
    
    # Neck
    print("\n[Neck]")
    net.add_layer( _get_RFB_s_block(from_layer=-8, in_filters=512, out_filters=512) ) # 21~41
    net.reserve_head()
    net.add_layer( _get_RFB_block(from_layer=-22, in_filters=1024, out_filters=1024) ) # 42~58
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=1024, out_filters=512) ) # 59~74
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=512, out_filters=256) ) # 75~90
    net.reserve_head()
    print("\nReserve head on {}".format(net.head_idx))

    # Head
    print("\n[Head]")
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-1]) + 1), filters=256, yolo_mask=[0,1,2], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-2]) + 1), filters=512, yolo_mask=[3,4,5], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-3]) + 1), filters=1024, yolo_mask=[6,7,8], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-4]) + 1), filters=512, yolo_mask=[9,10,11], num_classes=num_classes) )

    net.generate_config("RFB_vgg_300.cfg")


def Generate_RFB_vgg_512_r1(num_classes, augmentation, optimizer, save_dir):

    net = Net_generator(num_classes, augmentation, optimizer, save_dir, input_size=512)

    # Backbone
    print("\n[Backbone]")
    net.add_layer( _get_vgg_net() ) # 1~20
    net.reserve_head()
    
    # Neck
    print("\n[Neck]")
    net.add_layer( _get_RFB_block(in_filters=1024, out_filters=1024) )
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=1024, out_filters=512) )
    net.reserve_head()
    net.add_layer( _get_RFB_block(in_filters=512, out_filters=256) )
    net.reserve_head()
    print("\nReserve head on {}".format(net.head_idx))

    # Head
    print("\n[Head]")
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-1]) + 1), filters=256, yolo_mask=[0,1,2], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-2]) + 1), filters=512, yolo_mask=[3,4,5], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-3]) + 1), filters=1024, yolo_mask=[6,7,8], num_classes=num_classes) )
    net.add_layer( _get_head(from_layer=-(net.num_layers - int(net.head_idx[-4]) + 1), filters=1024, yolo_mask=[9,10,11], num_classes=num_classes) )

    net.generate_config("RFB_vgg_512_r1.cfg")


#------------------------------------------------------------------------------------------------------------#

if __name__=='__main__':

    save_dir = "./config/"
    num_classes = 1
    augmentation = YOLOImageAugmentation()
    optimizer = YOLOOptimizer(batch_size=64, subdivisions=16, learning_rate=0.0005, 
                            lr_decay_schedule={40000: 0.1, 45000:0.1}, max_batches=50000)

    Generate_RFB_vgg_512(num_classes, augmentation, optimizer, save_dir)
    Generate_RFB_vgg_300(num_classes, augmentation, optimizer, save_dir)
    Generate_RFB_vgg_512_r1(num_classes, augmentation, optimizer, save_dir)