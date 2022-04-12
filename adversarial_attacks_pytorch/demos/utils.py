import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.datasets as dsets

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (15, 20))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

def imshow_both(images, adv_images, title):
    # original과 aa 이미지를 받아서 동시에 출력
    original_images = images
    aa_images = adv_images

    np_original_images = original_images.numpy()
    np_aa_images = aa_images.numpy()

    np_original_images = np.transpose(np_original_images,(1,2,0))
    np_aa_images = np.transpose(np_aa_images,(1,2,0))

    np_original_aa_images = np.concatenate((np_original_images, np_aa_images), axis=0)

    fig = plt.figure(figsize = (15, 20))
    # plt.imshow(np_original_images)
    plt.imshow(np_original_aa_images)
    plt.title('predict: %s'%title)
    plt.show()
    
    return

    
def image_folder_custom_label(root, transform, idx2label) :
    
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2
