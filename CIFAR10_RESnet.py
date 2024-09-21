import torch.cuda
import tqdm
import torch.nn as nn
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(BasicBlock, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1)

        # Change self.c2 to take out_channel as input channels
        self.c2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=1)

        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x

        # First conv and batch norm
        x = self.c1(x)
        x = self.bn1(x)

        # Second conv and batch norm (corrected to use out_channel)
        x = self.c2(x)
        x = self.bn2(x)

        # Downsampling
        x_ = self.downsample(x_)
        x += x_
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, num_class = 10):
        super(ResNet, self).__init__()

        self.b1 = BasicBlock(in_channel=3, out_channel=64)
        self.b2 = BasicBlock(in_channel=64, out_channel=128)
        self.b3 = BasicBlock(in_channel=128, out_channel=256)

        self.pool = nn.AvgPool2d(kernel_size=2, padding=0)

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_class)

        self.relu = nn.ReLU()

    def forward(self, x):

        x= self.b1(x)
        x= self.pool(x)
        x= self.b2(x)
        x= self.pool(x)
        x= self.b3(x)
        x= self.pool(x)
        x = torch.flatten(x, start_dim=1)

        x= self.fc1(x)
        x= self.relu(x)
        x= self.fc2(x)
        x= self.relu(x)
        x= self.fc3(x)
        return x


transforms = Compose([
    RandomCrop((32,32), padding= 4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

training_data = CIFAR10(root="./", train=True, download= True, transform=transforms)
test_data = CIFAR10(root="./", train= False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet(num_class=10)
model.to(device)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(10):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()
        iterator.set_description(f"epoch:{epoch+1} , loss : {loss.item()}")
    torch.save(model.state_dict(),"ResNet.pth")

model.load_state_dict(torch.load("ResNet.pth", map_location=device))

num_corr= 0

with torch.no_grad():
    for data ,label in test_loader:

        output = model(data.to(device))
        pred = output.data.max(1)[1]
        corr = pred.eq(label.to(device).data).sum().item()
        num_corr += corr
    print(f"Accuracy: {num_corr/ len(test_data)}")