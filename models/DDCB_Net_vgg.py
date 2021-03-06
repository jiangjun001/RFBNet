import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, visual=1):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1_1 = nn.BatchNorm2d(inter_planes)
        self.conv1_1 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=(1, 1), dilation=1, bias=False)
        self.conv1_2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=visual, dilation=visual, bias=False)
        self.conv1_3 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=(2, 2), dilation=1, bias=False)
        self.conv1_4 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=visual, dilation=visual, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.bn2_1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=visual, dilation=visual, bias=False)
        self.droprate = dropRate
        self.visual = visual

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.visual == 2:
            out = self.conv1_1(self.relu(self.bn1_1(out)))
        if self.visual == 3:
            out = self.conv1_3(self.relu(self.bn1_1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.visual == 1:
            out = self.conv2(self.relu(self.bn2(out)))
        if self.visual == 2:
            out = self.conv1_2(self.relu(self.bn2_1(out)))
        if self.visual == 3:
            out = self.conv1_4(self.relu(self.bn2_1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DenseDDCB(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, growth_rate, block, dropRate=0.0):
        super(DenseDDCB, self).__init__()
        self.out_channels = out_planes
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
        self.in_planes = int(in_planes+nb_layers*growth_rate)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate, visual=i+1))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.relu1(self.conv1(self.relu1(self.bn1(self.layer(x)))))
    
class DenseDDCB_a(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, growth_rate, block, dropRate=0.0, stride=1):
        super(DenseDDCB_a, self).__init__()
        self.out_channels = out_planes 
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
        self.in_planes = int(in_planes+nb_layers*growth_rate)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_planes, self.out_channels, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.droprate = dropRate
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate, visual=i+1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(self.layer(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.relu1(out)
        return out
    


    
class DDCBNet(nn.Module):
    """DDCB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(DDCBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        block = BottleneckBlock
        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Norm = DenseDDCB(3, 512, 512, 12, block, dropRate=0.0)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
            
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def make_layers(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    bbl=block(inplanes, planes, stride, downsample)
    bbl.out_channels=planes*block.expansion
    layers.append(bbl)
    
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        bbl=block(inplanes, planes)
        bbl.out_channels=planes*block.expansion
        layers.append(bbl)
    return layers

def resnet(cfg,in_channel=3):
    layers = []
    layers += [nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    block= torchvision.models.resnet.Bottleneck
    layers += make_layers(block,64,64, cfg[0])
    layers += make_layers(block,64*block.expansion,128,cfg[1], stride=2)
    layers += make_layers(block,128*block.expansion,256,cfg[2], stride=2)
    layers += make_layers(block,256*block.expansion,512,cfg[3], stride=2)
    #layers += [nn.AvgPool2d(7, stride=1)]
    return layers


base = {
    'vgg_300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    'vgg_512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    'resnet-50':[3, 4, 6, 3],
}
    
def add_extras(size, cfg, i, nb_layers, grow_rate, block, dropRate, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    if size == 512:
        layers += [DenseDDCB(nb_layers, in_channels, cfg[0],grow_rate, block, dropRate)]
        layers += [DenseDDCB_a(nb_layers, cfg[0], cfg[1],grow_rate, block, dropRate, stride=2)]
        layers += [DenseDDCB_a(nb_layers, cfg[1], cfg[2],grow_rate, block, dropRate, stride=2)]
        layers += [DenseDDCB_a(nb_layers, cfg[2], cfg[3],grow_rate, block, dropRate, stride=2)]
        layers += [DenseDDCB_a(nb_layers, cfg[3], cfg[4],grow_rate, block, dropRate, stride=2)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [DenseDDCB(nb_layers, in_channels, cfg[0],grow_rate, block, dropRate)]
        layers += [DenseDDCB_a(nb_layers, cfg[0], cfg[1],grow_rate, block, dropRate, stride=2)]
        layers += [DenseDDCB_a(nb_layers, cfg[1], cfg[2],grow_rate, block, dropRate, stride=2)]
        layers += [BasicConv(cfg[2],128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,cfg[2],kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 512, 256],
    '512': [1024, 512, 256, 256, 256],
}

    
def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, model, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    if mode == "vgg_300" or mode =="vgg_512":
        return DDCBNet(phase, size, *multibox(size, vgg(base[model], 3),
                                   add_extras(size, extras[str(size)], 1024, 3, 12, BottleneckBlock, dropRate=0.0),
                                   mbox[str(size)], num_classes), num_classes)
    else:
        return DDCBNet(phase, size, *multibox(size, resnet(base[model], 3),
                                   add_extras(size, extras[str(size)], 1024, 3, 12, BottleneckBlock, dropRate=0.0),
                                   mbox[str(size)], num_classes), num_classes)
