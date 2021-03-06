import torch
import torch.nn as nn

"""
Implementation of VGG net (VGG11, VGG13, VGG16, VGG19) architecture
Simmonyan et al., 2014
(https://arxiv.org/abs/1409.1556)

Code written by Thiago Sotoriva Lermen
    2021-06-23 Initial commit
"""
VGG_types = {
    "VGG11": [
        64, "M",
        128, "M",
        256, 256, "M",
        512, 512, "M",
        512, 512, "M"
    ],
    "VGG13": [
        64, 64, "M",
        128, 128, "M",
        256, 256, "M",
        512, 512, "M",
        512, 512, "M" 
    ],
    "VGG16": [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M"
    ],
    "VGG19": [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M"
    ]
} 

class VGG_net(nn.Module): 
    def __init__(self, in_channels=3, num_classes=1000): 
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])
        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
      
  
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
          if type(x) == int:
            out_channles = x

            layers += [
              nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channles, 
                kernel_size=(3,3), 
                stride=(1,1), 
                padding=(1,1)
              ), 
              nn.ReLU()
            ]
            
            in_channels = x

          elif x == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = VGG_net(in_channels=3, num_classes=1000)
    print(model)
    x = torch.randn(3, 3, 224, 224)
    print(f'Model shape: {model(x).shape}')
