import torch 
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


class fcn8(nn.Module):
    def __init__(self, input_chan=3):
        super(fcn8,self).__init__()
        self.conv1 = nn.Conv2d(input_chan, 32, kernel_size=3)	#128x128x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)			#64x64x32
        self.conv3 = nn.Conv2d(64,128, kernel_size=3)			#32x32x128
        self.conv4 = nn.Conv2d(128,256 , kernel_size=3)			#16x16x256

        self.deconv4 = nn.ConvTranspose2D(256,128, stride=2, dilation=1, kernel_size=3, padding=1,output_padding=1)
        self.deconv3 = nn.ConvTranspose2D(128,64, stride=2, dilation=1, kernel_size=3, padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose2D(64,32, stride=2, dilation=1, kernel_size=3, padding=1,output_padding=1)
        self.deconv1 = nn.ConvTranspose2D(32,1, stride=2, dilation=1, kernel_size=3, padding=1,output_padding=1)
        
        self.final = nn.Conv2d(32,1,kernel_size=3)

        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self,x):
        batch = x.size[0]
        x = pool(bn1(F.relu(conv1(x))))
        x = pool(bn2(F.relu(conv2(x))))
        x = pool(bn3(F.relu(conv3(x))))
        x = pool(bn4(F.relu(conv4(x))))
        x = pool(bn41(F.relu(deconv4(x))))
        x = pool(bn3(F.relu(deconv3(x))))
        x = pool(bn2(F.relu(deconv2(x))))
        x = pool(bn1(F.relu(deconv1(x))))
        x = F.sigmoid(final(x))
       return x

model = fcn8()
print(model)