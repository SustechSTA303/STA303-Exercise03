import clip
# import torch
# import clip
# from torchvision.datasets import CIFAR100


# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# def model_inference(model, image, text_inputs):
    
#     with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)

#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#     labels, logits = similarity[0].topk(5)

#     return labels,logits

# image, class_id = cifar100[3637]
# image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)


# text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
# text_tokens = clip.tokenize(text_descriptions).cuda()
import os
import skimage
# import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import OrderedDict
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

#matplotlib inline
#config InlineBackend.figure_format = 'retina'

# images in skimage to use and their textual descriptions
descriptions = {
    #"page": "a page of text about segmentation",
    # "chelsea": "a facial photo of a tabby cat",
  

    
    # "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer",
    #test
    #"brick": "a road made of bricks",
    #"cell" : "an animal's cell",
    "chessboard_GRAY":" a chess board which isn't colorful",
    "coins":"a coin in the middle",
    "color":"a picture contains many colors",
    "grass":"a photo of green grass",
    "logo":"a special logo ",
    #"rocket": "a rocket standing on a launchpad",
    #"astronaut": "a portrait of an astronaut with the American flag",
    "motorcycle_right": "a red motorcycle standing in a garage",
    #"chelsea": "a facial photo of a tabby cat",
}

original_images = []
images = []
texts = []

# plt.figure(figsize=(16, 5))

# for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
#     name = os.path.splitext(filename)[0]
#     if name not in descriptions:
#         continue

#     image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
#     plt.subplot(2, 4, len(images) + 1)
#     plt.imshow(image)
#     plt.title(f"{filename}\n{descriptions[name]}")
#     plt.xticks([])
#     plt.yticks([])

#     original_images.append(image)
#     images.append(preprocess(image))
#     texts.append(descriptions[name])

# plt.tight_layout()
# plt.show()

for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")

# Define the required transformations
    preprocess = transforms.Compose([
    transforms.Resize(size=224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Lambda(lambda x: Image.merge("RGB", (x, x, x)) if x.mode != "RGB" else x),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    processed_image = preprocess(image)
    original_images.append(image)
    images.append(processed_image)
    texts.append(name)
    print(processed_image.shape)
    
# print(images)
# print(texts)
print("preprocess done")


image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()
print("feature done")
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
print("calculate done")
similarity = (100.0 * image_features.cpu() @ text_features.cpu().T).softmax(dim=-1)

count = len(descriptions)
print("count done")
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
# plt.colorbar()
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
  plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=20)
plt.show()
'''''
from torchvision.datasets import CIFAR100

cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
text_tokens = clip.tokenize(text_descriptions).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

top_probs, top_labels = similarity.cpu().topk(5, dim=-1)

plt.figure(figsize=(16, 16))

for i, image in enumerate(original_images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace=0.5)
plt.show()
'''''