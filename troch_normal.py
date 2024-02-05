import torch
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
training_data = CIFAR10(root="./", train = True, download = True, transform =ToTensor())
img = [item[0] for item in training_data]
img = torch.stack(img, dim=0).numpy()

mean_r = img[:,0,:,:].mean()
mean_g = img[:,1,:,:].mean()
mean_b = img[:,2,:,:].mean()
print(mean_r, mean_g, mean_b)

std_r = img[:,0,:,:].std()
std_g = img[:,1,:,:].std()
std_b = img[:,2,:,:].std()
print(std_r, std_g, std_b)