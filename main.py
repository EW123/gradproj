import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

traindir='/input/pl365/places365_standard/train'
valdir='/input/pl365/places365_standard/val'

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Conv2d(3,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		self.fc1=nn.Linear(16*54*54,14010)
		self.fc2=nn.Linear(14010,9807)
		self.fc3=nn.Linear(9807,365)


	def forward(self,x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.poll(F.relu(self.conv2(x)))
		x=x.view(-1,16*54*54)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc2(x)
		return x


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

	criterion=nn.CrossEntropyLoss()
	optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

	for epoch in range(2):

		running_loss=0.0
		for i,data in enumerate(trainloader,0):

			inputs,labels=data

			inputs,labels=Variable(inputs),Variables(labels)

			optimizer.zero_grad()
			
			

if __name__=='__main__':
	main()
