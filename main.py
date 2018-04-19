import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

traindir='/input/pl365/places365_standard/train'
valdir='/input/pl365/places365_standard/val'

class Net(nn.Module):
	def __init__(self):
		pass

	def forward(self,x):
		pass


def main():
	normalize=transforms.Normalize(mean=[0.485,0.456,0.406],
                                      std=[0.229,0.224,0.225])

	trainset=torchvision.datasets.ImageFolder(traindir,transform=transforms.Compose(
                [transforms.ToTensor(),
                 normalize,
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),]))
	
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
						shuffle=True,num_workers=2)

	valset=torchvision.datasets.ImageFolder(valdir,transforms.Compose([
				transforms.Scale(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
				]))
	
	valloader=torch.utils.data.DataLoader(valset,bach_size=4,
					      shuffle=False,num_workers=2)

	net=Net()

if __name__=='__main__':
	main()
