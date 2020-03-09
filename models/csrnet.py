import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from utils.models import make_layers

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.pred_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv_logvar = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self,x, aleatoric=False):
        size = x.size()
        x = self.frontend(x)
        x = self.backend(x)
        pred = self.pred_layer(x)
        pred = F.upsample(pred, size = size[2:])
        if aleatoric:
            logvar = self.conv_logvar(x)
            logvar = F.upsample(logvar, size = size[2:])
            logvar = F.softplus(logvar)
            return pred, logvar
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
