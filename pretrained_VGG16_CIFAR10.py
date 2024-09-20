import tqdm
import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor , Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
    Resize(224),
    RandomCrop((224,224), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914 , 0.4822,0.4465), std=(0.247, 0.243, 0.261))
])


training_data = CIFAR10(
    root = "./",
    train = True,
    download = True,
    transform = transforms
)

test_data = CIFAR10(
    root = "./",
    train = False,
    download = True,
    transform = transforms
)

training_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True)

fc =nn.Sequential(
    nn.Linear(512*7*7,4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096,4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096,10),
)

model.classifier = fc
model.to(device)

lr = 1e-4
optim = Adam(model.parameters(), lr = lr)

for epoch in range(30):
    iterator = tqdm.tqdm(training_loader)
    for data, label in iterator:
        optim.zero_grad()
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
    torch.save(model.state_dict(),"CIFAR_pretrained.pth")

model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

num_corr = 0

with torch.no_grad():
    for data , label in test_loader:
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        corr = pred.eq(label.to(device).data).sum().item()
        num_corr += corr
    print(f"Accuracy:{num_corr/len(test_data)}")
