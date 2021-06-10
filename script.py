import sys
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image



def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    
    image = image.unsqueeze(0)
    return image

data_transforms = torchvision.transforms.Compose([
       torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc=torch.nn.Linear(in_features=512, out_features=3)


resnet18.load_state_dict=torch.load('./resnet18-model.pt')
resnet18.eval()
output = resnet18((image_loader(data_transforms, sys.argv[1])))
print( np.argmax(resnet18(image_loader(data_transforms, sys.argv[1])).detach().numpy()))