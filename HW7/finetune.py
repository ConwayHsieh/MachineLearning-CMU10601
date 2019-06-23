# Conway Hsieh
# 10-601 - HW7
# 11/08/2018

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from skimage import io, transform

import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time, os, copy, PIL
import pandas as pd

data_directory = './hw7data/'
image_directory = './hw7data/images/'
testCSV = data_directory + 'test.csv'
trainCSV = data_directory + 'train.csv'
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
toPIL = transforms.ToPILImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LandmarksDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform):
		#load the csv file and store all the information
		self.CSV = pd.read_csv(csv_file)
		#print(self.CSV)
		self.root_dir = root_dir
		self.transform = transform
		self.IDs  = self.CSV.id
		try:
			self.landmark_id = self.CSV.landmark_id
			#print('Landmark IDs exist, so train dataset')
			self.Test = False
		except:
			#print('No Landmark IDs, so test dataset')
			self.Test = True
			pass

	def __len__(self):
		# Return length of entries
		return len(self.CSV)

	def __getitem__(self,idx):
		# Read the image of the index idx. Apply tranformation onto the image
		img_name = os.path.join(self.root_dir,
			self.IDs[idx] + '.jpg')
		image = io.imread(img_name)
		sample = image

		# Return the transformed image and its label
		if self.transform:
			sample = self.transform(toPIL(image))

		if self.Test:
			return sample 
		else:
			return sample, self.landmark_id[idx]

trainDataset = LandmarksDataset(csv_file = trainCSV, 
	root_dir = image_directory, 
	transform=transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		]))

#TEST = LandmarksDataset(csv_file = trainCSV, 
#	root_dir = image_directory, 
#	transform=None)

#fig = plt.figure()
#for i in range(len(trainDataset)):
#   a = TEST[i]
#   plt.imshow(a)
#   plt.pause(0.5)
#   sample = toPIL(trainDataset[21])
	#print(np.shape(trainDataset[i]))
	#print(sample)
	#print(type(sample))
#   plt.imshow(sample)
#   plt.pause(0.5)
#   if i == 1:
#       break

#for i in range(len(trainDataset)):
	#sample = trainDataset[i]
	#plt.imshow(sample)
	#plt.pause(0.15)
	#transformed_sample = resizeTsfm(sample)
	#plt.imshow(sample)
	#plt.pause(0.5)
#	if i == 3:
#	   break

#print('Hello World')
dataset_loader = torch.utils.data.DataLoader(trainDataset,
	batch_size=16, shuffle=True, num_workers=8)

def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# train 
		scheduler.step()
		model.train()

		running_loss = 0.0
		running_corrects = 0

		# iterate over data
		for inputs, labels in dataset_loader:
			#print(labels)
			inputs = inputs.to(device)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()
			#forward
			#track history 
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				_, preds = torch.max(outputs,1)
				loss = criterion(outputs,labels)
				#backwards
				loss.backward()
				optimizer.step()

			# statistics
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)

		dataset_sizes = len(dataset_loader)
		epoch_loss = running_loss / dataset_sizes
		epoch_acc = running_corrects.double()/ dataset_sizes

		print(' Loss: {:.4f} Acc: {:.4f}'.format(
			epoch_loss, epoch_acc))

		#deep copy model
		if epoch_acc > best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict())

	
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model

model_ft = models.resnet18(pretrained=True)
#print(model_ft)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
#print(model_ft)

#torch.save(model_ft.state_dict(), 'resnet18.pt')
#model_ft.load_state_dict(torch.load('resnet18.pt'))
#model_ft.eval()

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if __name__ == '__main__':
	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
		num_epochs=2)

torch.save(model_ft.state_dict(), 'resnet18_dict.pt')
torch.save(model_ft,'resnet18.pt')

input("Press Enter to continue ...")

testDataset = LandmarksDataset(csv_file = testCSV, root_dir = image_directory, 
	transform=transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		]))

test_loader = torch.utils.data.DataLoader(testDataset,
	batch_size=16, shuffle=False, num_workers=8)


#model_ft.load_state_dict(torch.load('resnet18_dict.pt'))
#model_ft = torch.load('resnet18.pt')
model_ft.eval()
model_ft.to(device)

#print(model_ft)

print('landmark_id', file=open("submission.txt", "a"))
#print(testDataset)
#print(len(testDataset))
#for i in range(len(testDataset)):
#	log_py = model_ft(testDataset[i])
#	pred = np.argmax(log_py.data.numpy(), axis=1)
#	print(pred, file=open("submission.txt", "a"))

if __name__ == '__main__':
	with torch.no_grad():
    	    for inputs in test_loader:
        	    outputs = model_ft.forward(inputs)
        	    #print(outputs)
        	    _, pred = torch.max(outputs, 1)
        	    for i in range(len(pred)):
        	    	print(pred[i].item(), file=open("submission.txt","a"))
        	    	print(pred[i].item())
        	    #print(torch.exp(_))
        	    #print(pred[0].item(),file=open("submission.txt","a"))


#print('DONE')