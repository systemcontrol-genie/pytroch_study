import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True)
fc = nn.Sequential(
    nn.Linear(512*7*7, 4096),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(4096, 10)
)
model.classifier = fc
model.to(device)

import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

# "transforms" 변수명을 수정하였습니다.
transforms = Compose([
    Resize(224),
    RandomCrop((224, 224)),  # 크롭 크기를 입력 이미지 크기와 일치하거나 작도록 수정하였습니다.
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

train_data = CIFAR10(root='./', train=True, download=True, transform=transforms)
test_data = CIFAR10(root='./', train=False, download=True, transform=transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()
        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CIFAR_PRETRAINED.pth")