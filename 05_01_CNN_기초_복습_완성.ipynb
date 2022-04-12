{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kcoDRkM1P2TF"
   },
   "source": [
    "# Convolutional Neural Networks (CNN)\n",
    "\n",
    "커널 기반의 Convolution 연산을 사용하여 이미지 처리에 적합한 네트워크\n",
    "\n",
    "![](https://miro.medium.com/max/1400/1*S7UzoZx2W2Wl75XKLjxPBg.jpeg)\n",
    "\n",
    "[용어 자세한 설명](http://taewan.kim/post/cnn/)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jxxz2XXoRFtG"
   },
   "source": [
    "## Convolution 연산\n",
    "\n",
    "![](https://miro.medium.com/max/780/1*k_Kw7RPCfMaIY7USoAedqA.gif)\n",
    "\n",
    "### 2D Convolution\n",
    "\n",
    "![](https://miro.medium.com/max/738/1*tmKo_Dvlf3bHCMFx_PKZYA.gif)\n",
    "\n",
    "## MaxPooling 연산\n",
    "\n",
    "![](https://miro.medium.com/max/1400/1*vOxthD0FpBR6fJcpPxq6Hg.gif)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uHPo9_eeWyNe"
   },
   "source": [
    "# 1. CNN을 구성하는 레이어"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "l109GAwWVyb4"
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JlKsHPXyR0xt"
   },
   "source": [
    "배치 크기 × 채널 × 높이(height) × 너비(widht)\n",
    "\n",
    "- 배치 크기 : 1\n",
    "- 채널 : 1 (그레이스케일 1, 컬러 3)\n",
    "- 높이(픽셀) : 28\n",
    "- 너비(픽셀) : 28"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[[0., 1., 2.],\n",
      "         [3., 4., 5.],\n",
      "         [6., 7., 8.]]])\n",
      "MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)\n",
      "tensor([[[4., 5.],\n",
      "         [7., 8.]]])\n",
      "AvgPool2d(kernel_size=2, stride=1, padding=0)\n",
      "tensor([[[2., 3.],\n",
      "         [5., 6.]]])\n"
     ]
    }
   ],
   "source": [
    "## Pooling example\n",
    "input_example = torch.tensor([[[0, 1.0, 2], [3, 4, 5], [6, 7, 8]]])\n",
    "print(input_example)\n",
    "# max pooling\n",
    "max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=1)\n",
    "print(max_pooling_layer)\n",
    "print(max_pooling_layer(input_example))\n",
    "# average pooling\n",
    "average_pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=1)\n",
    "print(average_pooling_layer)\n",
    "print(average_pooling_layer(input_example))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[[12., 20., 30.,  0.],\n",
      "         [20., 12.,  2.,  0.],\n",
      "         [ 0., 70.,  5.,  2.],\n",
      "         [ 8.,  2., 90.,  3.]]])\n",
      "\n",
      "max pooling:\n",
      " tensor([[[20., 30.],\n",
      "         [70., 90.]]])\n",
      "\n",
      "average pooling:\n",
      " tensor([[[16.,  8.],\n",
      "         [20., 25.]]])\n"
     ]
    }
   ],
   "source": [
    "## Quiz!!\n",
    "input_example = torch.tensor([[[12, 20, 30, 0.0], [20, 12, 2, 0], [0, 70, 5, 2], [8, 2, 90, 3]]])\n",
    "print(input_example)\n",
    "\n",
    "max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)\n",
    "print('\\nmax pooling:\\n', max_pooling_layer(input_example))\n",
    "\n",
    "average_pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)\n",
    "print('\\naverage pooling:\\n', average_pooling_layer(input_example))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "aI5Zgos0ScaA"
   },
   "source": [
    "## Input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "E4iRzurFWeNl",
    "outputId": "d28e9aab-d68a-41cc-f692-1c5be5f8f409"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([1, 1, 32, 32])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inputs = torch.Tensor(1, 1, 32, 32)\n",
    "\n",
    "inputs.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "IFu4-VJMSfVr"
   },
   "source": [
    "## 첫번째 Conv2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "LlxoKba8WfPE",
    "outputId": "a0336647-8f1d-40d1-dc52-92b985971d8f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n"
     ]
    }
   ],
   "source": [
    "conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)\n",
    "\n",
    "print(conv1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8SWZuPfpShvq"
   },
   "source": [
    "## 두번째 Conv2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "_alfJ48RWhEc",
    "outputId": "22a6fc83-df73-47dd-adde-2daf113eecab"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n"
     ]
    }
   ],
   "source": [
    "conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)\n",
    "\n",
    "print(conv2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_H63pzP3Sj1G"
   },
   "source": [
    "## MaxPooling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "TMnXrPXjWh7k",
    "outputId": "526f6e0f-fef8-41c2-a63f-f94c17fadc1c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n"
     ]
    }
   ],
   "source": [
    "pool = nn.MaxPool2d(kernel_size=2)\n",
    "\n",
    "print(pool)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TizdLtsdSmMD"
   },
   "source": [
    "## 넣어보기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "IiwzAfpfWi28",
    "outputId": "f01634f3-bc11-4ee4-cdcc-7ed642a748cc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 32, 32, 32])\n"
     ]
    }
   ],
   "source": [
    "out = conv1(inputs)\n",
    "\n",
    "print(out.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "qBlykNFmWni8",
    "outputId": "e19a7a3a-44ad-40e5-f5de-4b6a915d84a2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 32, 16, 16])\n"
     ]
    }
   ],
   "source": [
    "out = pool(out)\n",
    "\n",
    "print(out.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "nPXlBhl_WoWk",
    "outputId": "dc2f1e67-1402-4a7f-b336-5d34095092d5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 64, 16, 16])\n"
     ]
    }
   ],
   "source": [
    "out = conv2(out)\n",
    "\n",
    "print(out.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "UG9T1GH2WpJ0",
    "outputId": "e2d0ba84-02b6-42d6-a86d-3ae4266c55bc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 64, 8, 8])\n"
     ]
    }
   ],
   "source": [
    "out = pool(out)\n",
    "\n",
    "print(out.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OFnX7PUYS1wX"
   },
   "source": [
    "## Flatten\n",
    "\n",
    "첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "wZ-Kb7enWuVL",
    "outputId": "0e2498b4-fc80-43db-ba71-261c195a0e68"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 4096])\n"
     ]
    }
   ],
   "source": [
    "out = out.view(out.size(0), -1) \n",
    "\n",
    "print(out.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "E6RQFlt1S7pE"
   },
   "source": [
    "## Dense (Fully connected)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "pSbQkpGzWvVc",
    "outputId": "c49646ac-0bc4-42a4-bb41-62e6dccb8652"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 10])\n",
      "tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],\n",
      "       grad_fn=<AddmmBackward0>)\n"
     ]
    }
   ],
   "source": [
    "fc = nn.Linear(4096, 10)\n",
    "\n",
    "out = fc(out)\n",
    "\n",
    "print(out.shape)\n",
    "print(out)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HRO1c9lPWzp2"
   },
   "source": [
    "# 2. CNN으로 MNIST 분류하기\n",
    "\n",
    "Colab 에서 GPU 사용하기\n",
    "\n",
    "런타임 - 런타임 유형 변경 - 하드웨어 가속기 - GPU - 저장"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ljsTl2w8rewk",
    "outputId": "ef2924db-61ad-45f7-99bc-bfd3157dec9f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fri Apr  8 10:44:36 2022       \n",
      "+-----------------------------------------------------------------------------+\n",
      "| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |\n",
      "|-------------------------------+----------------------+----------------------+\n",
      "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
      "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
      "|                               |                      |               MIG M. |\n",
      "|===============================+======================+======================|\n",
      "|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |\n",
      "| N/A   34C    P8    26W / 149W |      0MiB / 11441MiB |      0%      Default |\n",
      "|                               |                      |                  N/A |\n",
      "+-------------------------------+----------------------+----------------------+\n",
      "                                                                               \n",
      "+-----------------------------------------------------------------------------+\n",
      "| Processes:                                                                  |\n",
      "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
      "|        ID   ID                                                   Usage      |\n",
      "|=============================================================================|\n",
      "|  No running processes found                                                 |\n",
      "+-----------------------------------------------------------------------------+\n"
     ]
    }
   ],
   "source": [
    "!nvidia-smi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "qZHU9DlUWwdL",
    "outputId": "aca208a8-cfb6-4dc0-cc34-d7831fa80d7e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torchsummary in /usr/local/lib/python3.7/dist-packages (1.5.1)\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torchvision.datasets as datasets\n",
    "import torchvision.transforms as transforms\n",
    "!pip install torchsummary\n",
    "from torchsummary import summary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pcvakI7-poNF"
   },
   "source": [
    "## GPU 사용 설정"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "uPGSbUCTW2Vk",
    "outputId": "4aff1b18-428a-4c1b-a3e8-2593cfd270b1"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cuda\n"
     ]
    }
   ],
   "source": [
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "\n",
    "# 랜덤 시드 고정\n",
    "torch.manual_seed(123)\n",
    "\n",
    "# GPU 사용 가능일 경우 랜덤 시드 고정\n",
    "if device == 'cuda':\n",
    "    torch.cuda.manual_seed_all(123)\n",
    "\n",
    "print(device)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "StbKmvigprMU"
   },
   "source": [
    "## Hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "SzJNTYbwW3RD"
   },
   "outputs": [],
   "source": [
    "learning_rate = 0.001\n",
    "training_epochs = 15\n",
    "batch_size = 64"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fijwZCCCpvJ9"
   },
   "source": [
    "## Dataset & Dataloader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "code_folding": [
     0
    ],
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 483,
     "referenced_widgets": [
      "4901899e3cf24858a30250f375632f35",
      "ffacf747a18b44a78753a4b7e461e3ba",
      "17197e1f18054a4baeed8a68b0f04ea9",
      "4b0a34422b2941f08d62bdadafe1db6a",
      "396d1863f7714b9db5b90751925871b8",
      "0b9b0612be2d403295f34cfd3f370c0c",
      "566a4198022644d48f82d4f627a6472b",
      "ab7b5da9ae0e45a487e4a5c2c42f1f8f",
      "55af8b2a44e944a38307c2af5ee20db8",
      "01907fde2dda4fb09d7c8db71a071457",
      "65d2ec0ecc1445018d00da85ebdf7429",
      "8e12fd76ac714a529854d733b1f506ea",
      "14816477a62e4fafa40641cc52fb838a",
      "ed087cdf81d449c895000a22c0017a31",
      "91af701e1dc442479536b053498b7035",
      "339c27944ae94df991b8afcc3fbef87f",
      "0d83cb72ba644e35876e93544d94ee16",
      "6903fcd5755142ab807db290cd923ba2",
      "3431aff4c7aa46ebafa9b52336ab7b0f",
      "af6ef42c6d0043658e6362f26977d07e",
      "46a952fecd004b77b3ab9cd5023dec6e",
      "be75483f089b48ff8cbdc4fe191d7d8d",
      "562b0f8fc58c4184a34102bb61935016",
      "12bc7d3bc1284c87a4ae697a95f54ec0",
      "fdf3477cfe2946cfb822488f18ecc228",
      "3a714f13813a4ac98183861bda092ca4",
      "2de158afa91d4d5495e13bcc561c771c",
      "5b993e4be5ae40a59b96c5edf36807e2",
      "db3f6aa4a888409283aeb2a40e2a6ad3",
      "2f3420134d7a4ef1b30898fab56d1ed3",
      "ede149f92c7747cc8dbc9c5eb2fb28f1",
      "2b49e611d29f4bc48e6c005247fa4128",
      "cb8821b727cb42a4a61df0cc719e8285",
      "92bef92388584f86862c574b187a7df0",
      "e48553321fb94e0d8c9f2edd35453a5f",
      "19f50be626814d189f21db81c8ab5c32",
      "baf581a9fa114e38b89c6baf34843386",
      "665b0c5d4fe2419a8d6758c190848070",
      "3801d303cdbd4651b6fc1607941e521d",
      "777c056da02b4ba392f509786e0f8ee5",
      "e13e7a56648d4a0a9a3116e2b4e8f0b6",
      "1279abfba8fc48beb1e9982552db5e65",
      "815e6ee3d34540eda9e50510677afbce",
      "6644bfddf7ac479a9f82ee11ee634aee"
     ]
    },
    "id": "myxbQFHMW5w7",
    "outputId": "62dde0ff-3156-4311-d475-d1737dfa8104"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4901899e3cf24858a30250f375632f35",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/9912422 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8e12fd76ac714a529854d733b1f506ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/28881 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "562b0f8fc58c4184a34102bb61935016",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/1648877 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw\n",
      "\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n",
      "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "92bef92388584f86862c574b187a7df0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/4542 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw\n",
      "\n",
      "input sahep : torch.Size([1, 28, 28]), output : 5\n"
     ]
    }
   ],
   "source": [
    "# transformation 정의하기\n",
    "data_transform = transforms.Compose([\n",
    "#             transforms.Resize((32, 32)),\n",
    "            transforms.ToTensor(),\n",
    "])\n",
    "\n",
    "mnist_train = datasets.MNIST(\n",
    "    root='MNIST_data/', # 다운로드 경로 지정\n",
    "    train=True, # True를 지정하면 훈련 데이터로 다운로드\n",
    "    transform=data_transform,\n",
    "    download=True)\n",
    "\n",
    "mnist_test = datasets.MNIST(\n",
    "    root='MNIST_data/', # 다운로드 경로 지정\n",
    "    train=False, # False를 지정하면 테스트 데이터로 다운로드\n",
    "    transform=data_transform,\n",
    "    download=True)\n",
    "\n",
    "data_loader = torch.utils.data.DataLoader(\n",
    "    dataset=mnist_train,\n",
    "    batch_size=batch_size,\n",
    "    shuffle=True,\n",
    "    drop_last=True)\n",
    "\n",
    "print('input sahep : %s, output : %s'%(mnist_train[0][0].shape, mnist_train[0][1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "OBwTwcJtxbZU"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "SsQJPL34xbZV"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "V0CYIdVYxbZV"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7qi1DvsFxbZV"
   },
   "source": [
    "# 1_CNN 실습"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4I0oWzh5xbZV"
   },
   "source": [
    "### pytorch에서 모델 선언하는 3가지 방법"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "code_folding": [
     0
    ],
    "id": "5b-4ni5exbZV"
   },
   "outputs": [],
   "source": [
    "## define CNN model\n",
    "from torch.autograd import Variable\n",
    "class CNN1(nn.Module):\n",
    "    def __init__(self): # input image = batch_size x 3 x 32 x 32\n",
    "        super(CNN1, self).__init__()\n",
    "        \n",
    "        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.maxpool = nn.MaxPool2d(2)\n",
    "\n",
    "    def forward(self, x):\n",
    "        out = self.conv(x)\n",
    "        out = self.relu(out)\n",
    "        out = self.maxpool(out)\n",
    "        return out  # input image = batch_size x 3 x 16 x 16\n",
    "\n",
    "    \n",
    "class CNN2(nn.Module):\n",
    "    def __init__(self): # input image = batch_size x 3 x 32 x 32\n",
    "        super(CNN2, self).__init__()\n",
    "\n",
    "        self.layer = nn.Sequential(\n",
    "            nn.Conv2d(3, 64, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2))\n",
    "\n",
    "    def forward(self, x):\n",
    "        out = self.layer(x)\n",
    "        return out  # input image = batch_size x 3 x 16 x 16      \n",
    "      \n",
    "\n",
    "class CNN3(nn.Module):\n",
    "    def __init__(self): # input image = batch_size x 3 x 32 x 32\n",
    "        super(CNN3, self).__init__()\n",
    "        layer = []\n",
    "        \n",
    "        layer.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))\n",
    "        layer.append(nn.ReLU())\n",
    "        layer.append(nn.MaxPool2d(2))\n",
    "        \n",
    "        self.layer = nn.Sequential(*layer)\n",
    "\n",
    "    def forward(self, x):\n",
    "        out = self.layer(x)\n",
    "        return out  # input image = batch_size x 3 x 16 x 16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "2Qv-tJBcxbZV"
   },
   "outputs": [],
   "source": [
    "sample_image = Variable(torch.zeros(64, 3, 32, 32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gMzrUZLnxbZV",
    "outputId": "a676f720-8205-4864-f250-c168dbda2109"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CNN1(\n",
      "  (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "  (relu): ReLU()\n",
      "  (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      ")\n",
      "torch.Size([64, 64, 16, 16])\n"
     ]
    }
   ],
   "source": [
    "cnn = CNN1()\n",
    "print(cnn)\n",
    "output1 = cnn(sample_image)\n",
    "print(output1.size())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "pD5mo1HqxbZW",
    "outputId": "60f502e1-3f28-4f90-e0b1-50db7dcb4f06"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CNN2(\n",
      "  (layer): Sequential(\n",
      "    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (1): ReLU()\n",
      "    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "  )\n",
      ")\n",
      "torch.Size([64, 64, 16, 16])\n"
     ]
    }
   ],
   "source": [
    "cnn = CNN2()\n",
    "print(cnn)\n",
    "output2 = cnn(sample_image)\n",
    "print(output2.size())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "MpeIynH-xbZW",
    "outputId": "e8ce76ec-c6cc-4bc5-ee37-3fb77ba310aa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CNN3(\n",
      "  (layer): Sequential(\n",
      "    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (1): ReLU()\n",
      "    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "  )\n",
      ")\n",
      "torch.Size([64, 64, 16, 16])\n"
     ]
    }
   ],
   "source": [
    "cnn = CNN3()\n",
    "print(cnn)\n",
    "output3 = cnn(sample_image)\n",
    "print(output3.size())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "j_qfOHMfxbZW"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "NMjA_85qxbZW"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "V2mVlEBxxbZW"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "Uj3ELNtCxbZW"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "IdjkefOUp1W2"
   },
   "source": [
    "# 네트워크 정의"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "code_folding": [
     0
    ],
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "NILp2IjjxbZW",
    "outputId": "40106493-064e-427a-e04d-798c66db5f18"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CNN_practice1(\n",
      "  (conv1): Conv2d(3, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "  (conv2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "  (conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n",
      "  (conv4): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "  (linear): Linear(in_features=4096, out_features=10, bias=True)\n",
      ")\n",
      "torch.Size([2, 4096])\n",
      "torch.Size([2, 10])\n",
      "----------------------------------------------------------------\n",
      "        Layer (type)               Output Shape         Param #\n",
      "================================================================\n",
      "            Conv2d-1          [-1, 512, 32, 32]          14,336\n",
      "            Conv2d-2          [-1, 256, 32, 32]       1,179,904\n",
      "            Conv2d-3          [-1, 256, 16, 16]         590,080\n",
      "            Conv2d-4           [-1, 16, 16, 16]          36,880\n",
      "            Linear-5                   [-1, 10]          40,970\n",
      "================================================================\n",
      "Total params: 1,862,170\n",
      "Trainable params: 1,862,170\n",
      "Non-trainable params: 0\n",
      "----------------------------------------------------------------\n",
      "Input size (MB): 0.01\n",
      "Forward/backward pass size (MB): 6.53\n",
      "Params size (MB): 7.10\n",
      "Estimated Total Size (MB): 13.65\n",
      "----------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "class CNN_practice1(nn.Module):\n",
    "    def __init__(self, n_classes=10): # input image = batch_size x 3 x 32 x 32\n",
    "        super(CNN_practice1, self).__init__()\n",
    "        \n",
    "        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)\n",
    "        \n",
    "        self.conv1 = nn.Conv2d(3, 512, 3, 1, 1)  # 512 x 32 x 32\n",
    "        self.conv2 = nn.Conv2d(512, 256, 3, 1, 1)   # 256 x 32 x 32\n",
    "        self.conv3 = nn.Conv2d(256, 256, 3, 2, 1)  # 256 x 16 x 16\n",
    "        self.conv4 = nn.Conv2d(256, 16, 3, 1, 1)  # 16 x 16 x 16 = 4096\n",
    "    \n",
    "        self.linear = nn.Linear(16*16*16, n_classes)  # 4096 -> 10\n",
    "        \n",
    "    def forward(self, x):\n",
    "        out=self.conv1(x)\n",
    "        \n",
    "        out=self.conv2(out)\n",
    "        \n",
    "        out=self.conv3(out)\n",
    "        \n",
    "        out=self.conv4(out)\n",
    "        \n",
    "        out = out.contiguous().view(-1, 256*4*4)\n",
    "\n",
    "        print(out.shape)\n",
    "        \n",
    "        out = self.linear(out)\n",
    "\n",
    "        print(out.shape)        \n",
    "        \n",
    "        \n",
    "        return out\n",
    "\n",
    "model1=CNN_practice1(n_classes=10).to(device)\n",
    "print(model1)\n",
    "summary(model1, input_size=(3, 32, 32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "code_folding": [
     0
    ],
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "DCSQS_2EW7h0",
    "outputId": "3c36a2a7-ccc7-4efa-e7dc-28a7ec47fef1"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CNN_practice2(\n",
      "  (layer1): Sequential(\n",
      "    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (1): ReLU()\n",
      "    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "  )\n",
      "  (layer2): Sequential(\n",
      "    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (1): ReLU()\n",
      "    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "  )\n",
      "  (fc): Linear(in_features=3136, out_features=10, bias=True)\n",
      ")\n",
      "----------------------------------------------------------------\n",
      "        Layer (type)               Output Shape         Param #\n",
      "================================================================\n",
      "            Conv2d-1           [-1, 32, 28, 28]             320\n",
      "              ReLU-2           [-1, 32, 28, 28]               0\n",
      "         MaxPool2d-3           [-1, 32, 14, 14]               0\n",
      "            Conv2d-4           [-1, 64, 14, 14]          18,496\n",
      "              ReLU-5           [-1, 64, 14, 14]               0\n",
      "         MaxPool2d-6             [-1, 64, 7, 7]               0\n",
      "            Linear-7                   [-1, 10]          31,370\n",
      "================================================================\n",
      "Total params: 50,186\n",
      "Trainable params: 50,186\n",
      "Non-trainable params: 0\n",
      "----------------------------------------------------------------\n",
      "Input size (MB): 0.00\n",
      "Forward/backward pass size (MB): 0.65\n",
      "Params size (MB): 0.19\n",
      "Estimated Total Size (MB): 0.84\n",
      "----------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "class CNN_practice2(torch.nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CNN_practice2, self).__init__()\n",
    "        # 첫번째층\n",
    "        # ImgIn shape=(?, 28, 28, 1)\n",
    "        #    Conv     -> (?, 28, 28, 32)\n",
    "        #    Pool     -> (?, 14, 14, 32)\n",
    "        self.layer1 = torch.nn.Sequential(\n",
    "            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),\n",
    "            torch.nn.ReLU(),\n",
    "            torch.nn.MaxPool2d(kernel_size=2, stride=2))\n",
    "\n",
    "        # 두번째층\n",
    "        # ImgIn shape=(?, 14, 14, 32)\n",
    "        #    Conv      ->(?, 14, 14, 64)\n",
    "        #    Pool      ->(?, 7, 7, 64)\n",
    "        self.layer2 = torch.nn.Sequential(\n",
    "            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),\n",
    "            torch.nn.ReLU(),\n",
    "            torch.nn.MaxPool2d(kernel_size=2, stride=2))\n",
    "\n",
    "        # 전결합층 7x7x64 inputs -> 10 outputs\n",
    "        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)\n",
    "\n",
    "    def forward(self, x):\n",
    "        out = self.layer1(x)\n",
    "        out = self.layer2(out)\n",
    "        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten\n",
    "        out = self.fc(out)\n",
    "        return out\n",
    "\n",
    "model2 = CNN_practice2().to(device)\n",
    "print(model2)\n",
    "summary(model2, input_size=(1, 28, 28))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "code_folding": [
     0
    ],
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "LKq1S2cuxbZX",
    "outputId": "69fdb7b6-c669-4550-a934-221226172bc5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LeNet5(\n",
      "  (softmax): Softmax(dim=1)\n",
      "  (feature_extractor): Sequential(\n",
      "    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))\n",
      "    (1): Tanh()\n",
      "    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)\n",
      "    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))\n",
      "    (4): Tanh()\n",
      "    (5): AvgPool2d(kernel_size=2, stride=2, padding=0)\n",
      "    (6): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))\n",
      "    (7): Tanh()\n",
      "  )\n",
      "  (classifier): Sequential(\n",
      "    (0): Linear(in_features=120, out_features=84, bias=True)\n",
      "    (1): Tanh()\n",
      "    (2): Linear(in_features=84, out_features=10, bias=True)\n",
      "  )\n",
      ")\n",
      "----------------------------------------------------------------\n",
      "        Layer (type)               Output Shape         Param #\n",
      "================================================================\n",
      "            Conv2d-1            [-1, 6, 28, 28]             156\n",
      "              Tanh-2            [-1, 6, 28, 28]               0\n",
      "         AvgPool2d-3            [-1, 6, 14, 14]               0\n",
      "            Conv2d-4           [-1, 16, 10, 10]           2,416\n",
      "              Tanh-5           [-1, 16, 10, 10]               0\n",
      "         AvgPool2d-6             [-1, 16, 5, 5]               0\n",
      "            Conv2d-7            [-1, 120, 1, 1]          48,120\n",
      "              Tanh-8            [-1, 120, 1, 1]               0\n",
      "            Linear-9                   [-1, 84]          10,164\n",
      "             Tanh-10                   [-1, 84]               0\n",
      "           Linear-11                   [-1, 10]             850\n",
      "          Softmax-12                   [-1, 10]               0\n",
      "================================================================\n",
      "Total params: 61,706\n",
      "Trainable params: 61,706\n",
      "Non-trainable params: 0\n",
      "----------------------------------------------------------------\n",
      "Input size (MB): 0.00\n",
      "Forward/backward pass size (MB): 0.11\n",
      "Params size (MB): 0.24\n",
      "Estimated Total Size (MB): 0.35\n",
      "----------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "## LeNet\n",
    "# https://deep-learning-study.tistory.com/368\n",
    "class LeNet5(nn.Module):\n",
    "    def __init__(self, n_classes):\n",
    "        super(LeNet5, self).__init__()\n",
    "        \n",
    "        self.softmax = nn.Softmax(dim=1)\n",
    "        \n",
    "        self.feature_extractor = nn.Sequential(            \n",
    "            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),\n",
    "            nn.Tanh(),\n",
    "            nn.AvgPool2d(kernel_size=2),\n",
    "            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),\n",
    "            nn.Tanh(),\n",
    "            nn.AvgPool2d(kernel_size=2),\n",
    "            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),\n",
    "            nn.Tanh()\n",
    "        )\n",
    "\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(in_features=120, out_features=84),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(in_features=84, out_features=n_classes),\n",
    "        )\n",
    "\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.feature_extractor(x)\n",
    "        x = torch.flatten(x, 1)\n",
    "        logits = self.classifier(x)\n",
    "        probs = self.softmax(logits)\n",
    "        return logits, probs\n",
    "\n",
    "N_CLASSES = 10\n",
    "model3 = LeNet5(N_CLASSES).to(device)\n",
    "print(model3)\n",
    "summary(model3, input_size=(1, 32, 32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "code_folding": [
     0,
     2
    ],
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "9B9-rQSlxbZX",
    "outputId": "78d68798-b28b-4917-d5a1-18c2aab0751c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AlexNet(\n",
      "  (features): Sequential(\n",
      "    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))\n",
      "    (1): ReLU(inplace=True)\n",
      "    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\n",
      "    (4): ReLU(inplace=True)\n",
      "    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (7): ReLU(inplace=True)\n",
      "    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (9): ReLU(inplace=True)\n",
      "    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (11): ReLU(inplace=True)\n",
      "    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "  )\n",
      "  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))\n",
      "  (classifier): Sequential(\n",
      "    (0): Dropout(p=0.5, inplace=False)\n",
      "    (1): Linear(in_features=9216, out_features=4096, bias=True)\n",
      "    (2): ReLU(inplace=True)\n",
      "    (3): Dropout(p=0.5, inplace=False)\n",
      "    (4): Linear(in_features=4096, out_features=4096, bias=True)\n",
      "    (5): ReLU(inplace=True)\n",
      "    (6): Linear(in_features=4096, out_features=1000, bias=True)\n",
      "  )\n",
      ")\n",
      "----------------------------------------------------------------\n",
      "        Layer (type)               Output Shape         Param #\n",
      "================================================================\n",
      "            Conv2d-1           [-1, 64, 56, 56]          23,296\n",
      "              ReLU-2           [-1, 64, 56, 56]               0\n",
      "         MaxPool2d-3           [-1, 64, 27, 27]               0\n",
      "            Conv2d-4          [-1, 192, 27, 27]         307,392\n",
      "              ReLU-5          [-1, 192, 27, 27]               0\n",
      "         MaxPool2d-6          [-1, 192, 13, 13]               0\n",
      "            Conv2d-7          [-1, 384, 13, 13]         663,936\n",
      "              ReLU-8          [-1, 384, 13, 13]               0\n",
      "            Conv2d-9          [-1, 256, 13, 13]         884,992\n",
      "             ReLU-10          [-1, 256, 13, 13]               0\n",
      "           Conv2d-11          [-1, 256, 13, 13]         590,080\n",
      "             ReLU-12          [-1, 256, 13, 13]               0\n",
      "        MaxPool2d-13            [-1, 256, 6, 6]               0\n",
      "AdaptiveAvgPool2d-14            [-1, 256, 6, 6]               0\n",
      "          Dropout-15                 [-1, 9216]               0\n",
      "           Linear-16                 [-1, 4096]      37,752,832\n",
      "             ReLU-17                 [-1, 4096]               0\n",
      "          Dropout-18                 [-1, 4096]               0\n",
      "           Linear-19                 [-1, 4096]      16,781,312\n",
      "             ReLU-20                 [-1, 4096]               0\n",
      "           Linear-21                 [-1, 1000]       4,097,000\n",
      "================================================================\n",
      "Total params: 61,100,840\n",
      "Trainable params: 61,100,840\n",
      "Non-trainable params: 0\n",
      "----------------------------------------------------------------\n",
      "Input size (MB): 0.59\n",
      "Forward/backward pass size (MB): 8.49\n",
      "Params size (MB): 233.08\n",
      "Estimated Total Size (MB): 242.16\n",
      "----------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "## AlexNet\n",
    "# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py\n",
    "class AlexNet(nn.Module):\n",
    "    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:\n",
    "        super().__init__()\n",
    "        self.features = nn.Sequential(\n",
    "            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(kernel_size=3, stride=2),\n",
    "            nn.Conv2d(64, 192, kernel_size=5, padding=2),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(kernel_size=3, stride=2),\n",
    "            nn.Conv2d(192, 384, kernel_size=3, padding=1),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Conv2d(384, 256, kernel_size=3, padding=1),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Conv2d(256, 256, kernel_size=3, padding=1),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(kernel_size=3, stride=2),\n",
    "        )\n",
    "        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Dropout(p=dropout),\n",
    "            nn.Linear(256 * 6 * 6, 4096),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Dropout(p=dropout),\n",
    "            nn.Linear(4096, 4096),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Linear(4096, num_classes),\n",
    "        )\n",
    "\n",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
    "        x = self.features(x)\n",
    "        x = self.avgpool(x)\n",
    "        x = torch.flatten(x, 1)\n",
    "        x = self.classifier(x)\n",
    "        return x\n",
    "    \n",
    "model4 = AlexNet(num_classes=1000).to(device)\n",
    "print(model4)\n",
    "summary(model4, input_size=(3, 227, 227))  # ImageNet의 입력사이즈 3X227X226"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "code_folding": [
     0
    ]
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 10])\n"
     ]
    }
   ],
   "source": [
    "## Residual Connection\n",
    "class ResBlock(nn.Module):\n",
    "    def __init__(self, ni):\n",
    "        super(ResBlock, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(ni, ni, 1)\n",
    "        self.conv2 = nn.Conv2d(ni, ni, 3, 1, 1)\n",
    "        self.classifier = nn.Linear(ni*24*24, 10)\n",
    "\n",
    "    def forward(self,x):\n",
    "        residual = x\n",
    "        out = F.relu(self.conv1(x))\n",
    "        out = F.relu(self.conv2(out))\n",
    "        \n",
    "        out += residual\n",
    "        \n",
    "        out = out.view(out.size(0),-1)\n",
    "        return self.classifier(out)\n",
    "\n",
    "block = ResBlock(3)\n",
    "x = torch.randn(1, 3, 24, 24)\n",
    "output = block(x)\n",
    "print(output.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "xZtIuAYhxbZX"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8aHu-ro-qPz8"
   },
   "source": [
    "## 로스, 옵티마이저 정의"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "id": "Cn2boAKTxbZX"
   },
   "outputs": [],
   "source": [
    "model = model2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "id": "j40CkSUpW9c7"
   },
   "outputs": [],
   "source": [
    "criterion = torch.nn.CrossEntropyLoss().to(device)\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "u6Kt0xgNW_T8",
    "outputId": "0b6a4eb5-c841-4737-dc68-1bd564098f31"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch:    1] loss = 0.179870427\n",
      "[Epoch:    2] loss = 0.055393476\n",
      "[Epoch:    3] loss = 0.0411735997\n",
      "[Epoch:    4] loss = 0.0330234393\n",
      "[Epoch:    5] loss = 0.0270446781\n",
      "[Epoch:    6] loss = 0.0224145446\n",
      "[Epoch:    7] loss = 0.0180260204\n",
      "[Epoch:    8] loss = 0.0151498085\n",
      "[Epoch:    9] loss = 0.0125806704\n",
      "[Epoch:   10] loss = 0.0104894089\n",
      "[Epoch:   11] loss = 0.00882262923\n",
      "[Epoch:   12] loss = 0.0080840718\n",
      "[Epoch:   13] loss = 0.00627558259\n",
      "[Epoch:   14] loss = 0.00570008019\n",
      "[Epoch:   15] loss = 0.00586899649\n"
     ]
    }
   ],
   "source": [
    "for epoch in range(training_epochs):\n",
    "    avg_loss = 0\n",
    "\n",
    "    for x_train, y_train in data_loader:\n",
    "        x_train = x_train.to(device)\n",
    "        y_train = y_train.to(device)\n",
    "\n",
    "        y_pred = model(x_train)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(y_pred, y_train)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        avg_loss += loss / len(data_loader) # loss / 배치의 갯수\n",
    "\n",
    "    print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "NvZkY1i6XATz",
    "outputId": "9c36ea08-9a76-472b-fcd5-77a80f86c6ec"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9840999841690063\n"
     ]
    }
   ],
   "source": [
    "with torch.no_grad(): # Gradient를 업데이트하지 않는다\n",
    "    x_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)\n",
    "    y_test = mnist_test.targets.to(device)\n",
    "\n",
    "    y_pred = model(x_test)\n",
    "    correct_prediction = torch.argmax(y_pred, 1) == y_test\n",
    "    accuracy = correct_prediction.float().mean()\n",
    "\n",
    "    print('Accuracy:', accuracy.item())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cyTmoithrzoP"
   },
   "source": [
    "# (숙제) 정확도 99%로 만들어보자!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "YjXtDKGpxbZY"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "jdEIzPk-9pT9"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "machine_shape": "hm",
   "name": "05-01_CNN 기초_복습_완성.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "01907fde2dda4fb09d7c8db71a071457": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "0b9b0612be2d403295f34cfd3f370c0c": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "0d83cb72ba644e35876e93544d94ee16": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "1279abfba8fc48beb1e9982552db5e65": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "12bc7d3bc1284c87a4ae697a95f54ec0": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_5b993e4be5ae40a59b96c5edf36807e2",
      "placeholder": "​",
      "style": "IPY_MODEL_db3f6aa4a888409283aeb2a40e2a6ad3",
      "value": ""
     }
    },
    "14816477a62e4fafa40641cc52fb838a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_0d83cb72ba644e35876e93544d94ee16",
      "placeholder": "​",
      "style": "IPY_MODEL_6903fcd5755142ab807db290cd923ba2",
      "value": ""
     }
    },
    "17197e1f18054a4baeed8a68b0f04ea9": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_ab7b5da9ae0e45a487e4a5c2c42f1f8f",
      "max": 9912422,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_55af8b2a44e944a38307c2af5ee20db8",
      "value": 9912422
     }
    },
    "19f50be626814d189f21db81c8ab5c32": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_e13e7a56648d4a0a9a3116e2b4e8f0b6",
      "max": 4542,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_1279abfba8fc48beb1e9982552db5e65",
      "value": 4542
     }
    },
    "2b49e611d29f4bc48e6c005247fa4128": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "2de158afa91d4d5495e13bcc561c771c": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "2f3420134d7a4ef1b30898fab56d1ed3": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "339c27944ae94df991b8afcc3fbef87f": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "3431aff4c7aa46ebafa9b52336ab7b0f": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "3801d303cdbd4651b6fc1607941e521d": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "396d1863f7714b9db5b90751925871b8": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "3a714f13813a4ac98183861bda092ca4": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_2b49e611d29f4bc48e6c005247fa4128",
      "placeholder": "​",
      "style": "IPY_MODEL_cb8821b727cb42a4a61df0cc719e8285",
      "value": " 1649664/? [00:00&lt;00:00, 19774827.35it/s]"
     }
    },
    "46a952fecd004b77b3ab9cd5023dec6e": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "4901899e3cf24858a30250f375632f35": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_ffacf747a18b44a78753a4b7e461e3ba",
       "IPY_MODEL_17197e1f18054a4baeed8a68b0f04ea9",
       "IPY_MODEL_4b0a34422b2941f08d62bdadafe1db6a"
      ],
      "layout": "IPY_MODEL_396d1863f7714b9db5b90751925871b8"
     }
    },
    "4b0a34422b2941f08d62bdadafe1db6a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_01907fde2dda4fb09d7c8db71a071457",
      "placeholder": "​",
      "style": "IPY_MODEL_65d2ec0ecc1445018d00da85ebdf7429",
      "value": " 9913344/? [00:00&lt;00:00, 3648028.15it/s]"
     }
    },
    "55af8b2a44e944a38307c2af5ee20db8": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "562b0f8fc58c4184a34102bb61935016": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_12bc7d3bc1284c87a4ae697a95f54ec0",
       "IPY_MODEL_fdf3477cfe2946cfb822488f18ecc228",
       "IPY_MODEL_3a714f13813a4ac98183861bda092ca4"
      ],
      "layout": "IPY_MODEL_2de158afa91d4d5495e13bcc561c771c"
     }
    },
    "566a4198022644d48f82d4f627a6472b": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "5b993e4be5ae40a59b96c5edf36807e2": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "65d2ec0ecc1445018d00da85ebdf7429": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "6644bfddf7ac479a9f82ee11ee634aee": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "665b0c5d4fe2419a8d6758c190848070": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "6903fcd5755142ab807db290cd923ba2": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "777c056da02b4ba392f509786e0f8ee5": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "815e6ee3d34540eda9e50510677afbce": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "8e12fd76ac714a529854d733b1f506ea": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_14816477a62e4fafa40641cc52fb838a",
       "IPY_MODEL_ed087cdf81d449c895000a22c0017a31",
       "IPY_MODEL_91af701e1dc442479536b053498b7035"
      ],
      "layout": "IPY_MODEL_339c27944ae94df991b8afcc3fbef87f"
     }
    },
    "91af701e1dc442479536b053498b7035": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_46a952fecd004b77b3ab9cd5023dec6e",
      "placeholder": "​",
      "style": "IPY_MODEL_be75483f089b48ff8cbdc4fe191d7d8d",
      "value": " 29696/? [00:00&lt;00:00, 391423.38it/s]"
     }
    },
    "92bef92388584f86862c574b187a7df0": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_e48553321fb94e0d8c9f2edd35453a5f",
       "IPY_MODEL_19f50be626814d189f21db81c8ab5c32",
       "IPY_MODEL_baf581a9fa114e38b89c6baf34843386"
      ],
      "layout": "IPY_MODEL_665b0c5d4fe2419a8d6758c190848070"
     }
    },
    "ab7b5da9ae0e45a487e4a5c2c42f1f8f": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "af6ef42c6d0043658e6362f26977d07e": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "baf581a9fa114e38b89c6baf34843386": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_815e6ee3d34540eda9e50510677afbce",
      "placeholder": "​",
      "style": "IPY_MODEL_6644bfddf7ac479a9f82ee11ee634aee",
      "value": " 5120/? [00:00&lt;00:00, 144068.41it/s]"
     }
    },
    "be75483f089b48ff8cbdc4fe191d7d8d": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "cb8821b727cb42a4a61df0cc719e8285": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "db3f6aa4a888409283aeb2a40e2a6ad3": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "e13e7a56648d4a0a9a3116e2b4e8f0b6": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "e48553321fb94e0d8c9f2edd35453a5f": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_3801d303cdbd4651b6fc1607941e521d",
      "placeholder": "​",
      "style": "IPY_MODEL_777c056da02b4ba392f509786e0f8ee5",
      "value": ""
     }
    },
    "ed087cdf81d449c895000a22c0017a31": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_3431aff4c7aa46ebafa9b52336ab7b0f",
      "max": 28881,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_af6ef42c6d0043658e6362f26977d07e",
      "value": 28881
     }
    },
    "ede149f92c7747cc8dbc9c5eb2fb28f1": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "fdf3477cfe2946cfb822488f18ecc228": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_2f3420134d7a4ef1b30898fab56d1ed3",
      "max": 1648877,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_ede149f92c7747cc8dbc9c5eb2fb28f1",
      "value": 1648877
     }
    },
    "ffacf747a18b44a78753a4b7e461e3ba": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_0b9b0612be2d403295f34cfd3f370c0c",
      "placeholder": "​",
      "style": "IPY_MODEL_566a4198022644d48f82d4f627a6472b",
      "value": ""
     }
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
