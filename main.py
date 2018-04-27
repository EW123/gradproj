import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.autograd import Variable

traindir='/input/pl365/places365_standard/train'
valdir='/input/pl365/places365_standard/val'


def main():
	print("enter main function")
	normalize=transforms.Normalize(mean=[0.485,0.456,0.406],
                                      std=[0.229,0.224,0.225])

	trainset=torchvision.datasets.ImageFolder(traindir,transform=transforms.Compose(
                [transforms.RandomResizedCrop(224),
            	transforms.RandomHorizontalFlip(),
            	transforms.ToTensor(),
            	normalize,]))
	print("prepare trainloader")
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
						shuffle=True,num_workers=2)

	valset=torchvision.datasets.ImageFolder(valdir,transforms.Compose([
				            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))
	print("prepare valloader")
	valloader=torch.utils.data.DataLoader(valset,batch_size=4,
					      shuffle=False,num_workers=2)

	net=models.resnet18().cuda()

	criterion=nn.CrossEntropyLoss().cuda()
	optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

	for epoch in range(2):

		print("epoch %d" %epoch)
		running_loss=0.0
		for i,data in enumerate(trainloader,0):

			inputs,labels=data
			inputs=inputs.cuda(async=True)
			labels=labels.cuda(async=True)

			inputs,labels=Variable(inputs),Variable(labels)

			optimizer.zero_grad()

			outputs=net(inputs)
			
			loss=criterion(outputs,labels)

			loss.backward()

			optimizer.step()
			
			running_loss+=loss.data[0]
			if i%2000==0:
				print("[%d %d] loss:%.3f" %(epoch,i,running_loss))

			

if __name__=='__main__':
	main()
