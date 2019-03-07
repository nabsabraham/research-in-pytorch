'''
https://stackoverflow.com/questions/44146655/how-to-convert-pretrained-fc-layers-to-conv-layers-in-pytorch#44410334

converting fc to conv layers or just how to remove them
'''
import torch
import torch.nn as nn
from torchvision import models

# 1. LOAD PRE-TRAINED VGG16
model = models.vgg16(pretrained=True)

# 2. GET CONV LAYERS
features = model.features

# 3. GET FULLY CONNECTED LAYERS
fcLayers = nn.Sequential(
    # stop at last layer
    *list(model.classifier.children())[:-1]
)

# 4. CONVERT FULLY CONNECTED LAYERS TO CONVOLUTIONAL LAYERS

### convert first fc layer to conv layer with 512x7x7 kernel
fc = fcLayers[0].state_dict()
in_ch = 512
out_ch = fc["weight"].size(0)

firstConv = nn.Conv2d(in_ch, out_ch, 7, 7)

### get the weights from the fc layer
firstConv.load_state_dict({"weight":fc["weight"].view(out_ch, in_ch, 7, 7),
                           "bias":fc["bias"]})

# CREATE A LIST OF CONVS
convList = [firstConv]

# Similarly convert the remaining linear layers to conv layers 
for layer in enumerate(fcLayers[1:]):
    if isinstance(module, nn.Linear):
        # Convert the nn.Linear to nn.Conv
        fc = module.state_dict()
        in_ch = fc["weight"].size(1)
        out_ch = fc["weight"].size(0)
        conv = nn.Conv2d(in_ch, out_ch, 1, 1)

        conv.load_state_dict({"weight":fc["weight"].view(out_ch, in_ch, 1, 1),
            "bias":fc["bias"]})

        convList += [conv]
    else:
        # Append other layers such as ReLU and Dropout
        convList += [layer]

# Set the conv layers as a nn.Sequential module
convLayers = nn.Sequential(*convList)  