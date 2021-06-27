import torch
import torch.nn as nn


# AlexNet architecture
#   Input: 3x224x224
#   Output: 1000
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),  # (96 x 55 x 55)
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2), # section 3.3
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(96, 256, 5, padding=2), # 256 x 27 x 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), # 256 x 13 x 13
            nn.Conv2d(256, 384, 3, padding=1), # 384 x 13 x 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), # 384 x 13 x 13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), # 256 x 13 x 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # 256 x 6 x 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = AlexNet()
    print(net)