import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import OrderedDict
import torch
import clip
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
testing_loss = []
testing_num = 0
testing_right = 0

with torch.no_grad():
    model.eval()

    for index in range(len(cifar100)):
        testing_num += 1
        print(f"{testing_num/100:.2f}%")
        if testing_num >= 100:
            break
        image, label = cifar100[index]
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        value, predict = similarity[0].topk(1)
        testing_loss.append(value.item()*0.01)
        if predict == label:
            testing_right +=1
        
        else:
            print(str(predict) ,"and",str(label) )



    average_loss = sum(testing_loss) /testing_num
    average_acc = testing_right / testing_num

    print(f"Average testing loss: {average_loss:.4f}, average testing accuracy: {average_acc*100:.2f}%")
