## 220419, 코로나 검사모델, wygo

## 설치
# pip install streamlit

##실행
# streamlit run 06_03_advanced_cnn_CAM_AA_220412_final_president_serving.py

## load ibrary
import streamlit as st
import numpy as np
import json
import pprint
from PIL import Image
import PIL.Image as pilimg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from sklearn.metrics import f1_score
from tqdm import tqdm


st.set_page_config(page_title="Covid19 detection", page_icon="🤖")
st.title('President detection')
st.header('KAERI 인공지능 미니석사과정 6주차 실습')
st.markdown("[Reference](https://keep-steady.tistory.com/35)")



## parameter
is_Test = False
# is_Test = True
num_epochs = 30
batch_size  = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset = 'covid'
dataset = 'president'

if dataset == 'president':
    data_path = 'data_president/data_president'  # 데이터 경로, 이 안엔 CLASS가 폴더별로 정리
    label_path = 'data_president/data_president/president_label.json'
    
    with open(label_path) as json_file:
        class_names = json.load(json_file)
        pprint.pprint(class_names)

        
elif dataset == 'covid':
    data_path = 'data_covid19'  # 데이터 경로, 이 안엔 CLASS가 폴더별로 정리
    label_path = 'data_covid19/president_label.json'
    
    class_names={
        '0': 'anomal',
        '1': 'normal'        
    }
    
save_path='output_%s'%(dataset)
num_classes = len(class_names)
print('num class : %d'%(num_classes))

## For test
weights_path = os.path.join(save_path, 'best_model.pt')



## load model
from efficientnet_pytorch import EfficientNet
@st.cache(allow_output_mutation=True)
## load model for test
def load_model_for_test(weights_path, num_classes=2):
    
    # load best model from weight
    # weights_path = 'output_crop/model_4_100.00_100.00.pt'

    
    ## load model
    from efficientnet_pytorch import EfficientNet
    model_name = 'efficientnet-b0'  # b5
    # num_classes = 2  # 장싱, 비정상
    freeze_extractor = True  # FC layer만 학습하고 efficientNet extractor 부분은 freeze하여 학습시간 단축, 89860 vs 4097408
    use_multi_gpu = True

    model_load = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)  # load weight
    model_load.load_state_dict(state_dict, strict=False)  # insert weight to model structure

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)            
    print('학습 parameters 개수 : %d'%(count_parameters(model_load)))

    # multi gpu(2개 이상)를 사용하는 경우
    if use_multi_gpu:
        num_gpu = torch.cuda.device_count()
        if (device.type=='cuda') and (num_gpu > 1):
            print('use multi gpu : %d' % (num_gpu))
            model_load = nn.DataParallel(model_load, device_ids=list(range(num_gpu)))

    model_load = model_load.to(device)


    # define optimizer, criterion
    criterion = nn.CrossEntropyLoss()  # 분류이므로 cross entrophy 사용    
    
    return model_load, criterion, device


model_load, criterion, device = load_model_for_test(weights_path, num_classes=num_classes)


## efficient net
print('!!!!!! load done, efficient net!!')
# features_fn  = model_load.module.features_fn
# classifier_fn= model_load.module.classifier_fn
features_fn  = model_load.features_fn
classifier_fn= model_load.classifier_fn


def GradCAM(img, class_idx, features_fn, classifier_fn):

    feature = features_fn(img.to(device))  # A, [1, 1280, 7, 7]
    _, N, H, W = feature.size()  # 1280, 7, 7
    out = classifier_fn(feature)  # shape : [1, 7] - [-2.6593, -5.3088, -2.2299, -1.2445,  2.3712, -2.7554, 11.4827]
    class_score = out[0, class_idx]  # class_idxc=6이면 심상정일 때의 score, 11.4827

    # gradients via back-propagation
    # 특정 클래스(class_idx)의 gradient (dy/dA)
    # 최종단과 feature단의 미분값을 구한다
    grads = torch.autograd.grad(class_score, feature)  # grads = K.gradients(y_c, layer_output)[0]

    # a_k_c, Global average pooling
    weights = torch.mean(grads[0][0], axis=(1,2))  # (1280, 7, 7) -> (1280)

    ####################################################
    ## 1. torch 방법
    heatmap = torch.matmul(weights, feature.view(N, H*W))  # liniear combination : a_k_c * A__k, 1280 * (1280, 49)
    heatmap = heatmap.view(H, W).cpu().detach().numpy()  # (7, 7)
    ####################################################
    ## 2. 다른 방법 - https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
    # target = feature[0].cpu().data.numpy()
    # heatmap = np.zeros(target.shape[1:], dtype=np.float32)
    # for i, w in enumerate(weights):
    #     heatmap += w.cpu().data.numpy() * target[i, :, :]
    ####################################################
    grad_cam = np.maximum(heatmap, 0)  # ReLU, 0보다 큰값만 살린다
    # 0~1 사이로 정규화
    grad_cam -= grad_cam.min()
    grad_cam /= grad_cam.max()
    
    return grad_cam


# Opens image from disk, normalizes it and converts to tensor
read_image_to_tensor = transforms.Compose([
    lambda x: Image.open(x),
    lambda x: x.convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])


## Plot GradCAM
import glob
imshow_number = 3
top_k = 5

def plot_gradcam(data_path, class_label=0, top_k=3, imshow_number=5):
#     class_label = 0  # 0: 이재명, ..
    if top_k > num_classes:
        top_k = num_classes

    img_path = data_path

    img_tensor = read_image_to_tensor(img_path)
    output = model_load(img_tensor.to(device))  # tensor([[-2.6664,  2.5967]])
    output_softmax = nn.Softmax(dim=1)(output)  # tensor([[0.0094, 0.9906]]
    class_probability, class_idx = torch.topk(output_softmax, top_k)  # top1: (tensor([[0.9944]]), tensor([[1]])), tok2라면 ([[0.9819, 0.0181]], [[1, 0]])
    pp, cc = torch.topk(nn.Softmax(dim=1)(model_load(img_tensor.to(device))), 1)  # 3->1

    class_label_string = class_names['%s'%(class_label)]
    class_idx_string = class_names['%s'%(class_idx.cpu().numpy()[0][0])]
    class_probability_string = class_probability.cpu().tolist()[0][0]

    print('%s: %s -> %s (%.2f)'%(('True' if class_label==class_idx.cpu().numpy()[0][0] else '!!!Fail!!!'), class_label_string, class_idx_string, class_probability_string*100))

    result = []
    # plot GradCAM
    plt.figure(figsize=(15, 5))
    for i, (p, c) in enumerate(zip(class_probability[0], class_idx[0])):
        plt.subplot(1, top_k, i+1)
        grad_cam = GradCAM(img_tensor, int(c), features_fn, classifier_fn)  # 7x7
        img = Image.open(img_path)
        grad_cam = Image.fromarray(grad_cam)
        grad_cam = grad_cam.resize(img.size, resample=Image.LINEAR)  # 7x7 -> 224x224
        # print(i, p, c, str(int(c.cpu())))
        plt.title('{}: {:.1f}%'.format(class_names[str(int(c.cpu()))], 100*float(p)))
        plt.axis('off')
        plt.imshow(img)
        plt.imshow(np.array(grad_cam), alpha=0.3, cmap='jet')        
        
        result.append('{}: {:.1f}%'.format(class_names[str(int(c.cpu()))], 100*float(p)))

    # plt.show()
    plt.savefig('tmp.jpg')
    return result



uploaded_file = st.file_uploader('Upload president image', type=['jpg', 'png'])
print(uploaded_file)

if uploaded_file:
    # Using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')

    data_path = uploaded_file
    result = plot_gradcam(data_path, class_label=0, top_k=top_k, imshow_number=imshow_number)

    # Using PIL
    image = Image.open('tmp.jpg')
    st.header('President 판독 결과: %s'%(result[0]).replace('YSR', '윤석열').replace('LNY', '이낙연').replace('ACS', '안철수').replace('LJS', '이준석').replace('HJP', '홍준표').replace('YSR', '윤석열'))
    st.image(image, caption='Analyzed Image')
    