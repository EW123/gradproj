import torch
import torchvision
import torchvision.transforms as transforms

traindir='/input/pl365'
def main():
	transform=transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize(mean=[0.485,0.456,0.406],
				      std=[0.229,0.224,0.225]),
		 transforms.RandomSizedCrop(224),
		 transforms.RandomHorizontalFlip(),])

	trainset=torchvision.datasets.ImageFolder(traindir,transform)
	
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
						shuffle=True,numworkers=2)



if __name__=='__main__':
	main()
