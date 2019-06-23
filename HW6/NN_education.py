# Conway Hsieh
# NN for Education Data

import sys, csv, math, string, numpy as np

# import written functions
from func import sigmoid, normalize

vectSigmoid = np.vectorize(sigmoid)

stepSize = 0.4

#filename is input 1,2,3,4,5
trainDataFileName = sys.argv[1]
trainKeyName = sys.argv[2]
devFileName = sys.argv[3]
devWeight1FileName = sys.argv[4]
devWeight2FileName = sys.argv[5]

# convert CSV contents into a list of lists
trainDataList = list();
with open(trainDataFileName) as inputFile1:
    readCSV = csv.reader(inputFile1, delimiter=',')
    for row in readCSV:
    	trainDataList.append(row)

# first row is the name of attr, should be same for both test and train
labelList = trainDataList[0]
# delete first row to keep only data points
del trainDataList[0]

#print(labelList)
#print(trainDataList)

trainKeyList = list();
with open(trainKeyName) as inputFile2:
    readCSV = csv.reader(inputFile2, delimiter=',')
    for row in readCSV:
    	trainKeyList.append(row)
#print(trainKeyList)

devDataList = list();
with open(devFileName) as inputFile3:
    readCSV = csv.reader(inputFile3, delimiter=',')
    for row in readCSV:
    	devDataList.append(row)
# delete first row to keep only data points; attribute labels same as training
del devDataList[0]
#print(devDataList)

devWeight1List = list();
with open(devWeight1FileName) as inputFile4:
    readCSV = csv.reader(inputFile4, delimiter=',')
    for row in readCSV:
    	devWeight1List.append(row)
devWeight1Array = np.asarray(devWeight1List).T
devWeight1Array = devWeight1Array.astype(np.float)
#print(devWeight1Array)

devWeight2List = list();
with open(devWeight2FileName) as inputFile5:
    readCSV = csv.reader(inputFile5, delimiter=',')
    for row in readCSV:
    	devWeight2List.append(row)
devWeight2Array = np.asarray(devWeight2List)
devWeight2Array = devWeight2Array.T
devWeight2Array = devWeight2Array.astype(np.float)
#print(devWeight2Array)

# Convert Data Arrays into numpy arrays, normalized between 0,1
gradeMax = 100
gradeMin = 0

numTrainData = len(trainDataList)
numDevData = len(devDataList)
numAttr = len(labelList)
#print(numTrainData)
#print(numDevData)
#print(numAttr)

trainDataArray = np.zeros((numTrainData,numAttr))
trainKeyArray = np.zeros((numTrainData,1))
devDataArray = np.zeros((numDevData,numAttr))
#print(trainDataArray)
#print(devDataArray)
#print(trainDataList[0][0])

for i in range(numTrainData):
	trainDataArray[i,0] = normalize(trainDataList[i][0], gradeMax, gradeMin)
	trainDataArray[i,1] = normalize(trainDataList[i][1], gradeMax, gradeMin)
	trainDataArray[i,2] = normalize(trainDataList[i][2], gradeMax, gradeMin)
	trainDataArray[i,3] = normalize(trainDataList[i][3], gradeMax, gradeMin)
	trainDataArray[i,4] = normalize(trainDataList[i][4], gradeMax, gradeMin)

for i in range(numTrainData):
		trainKeyArray[i,0] = float(trainKeyList[i][0])/100

for i in range(numDevData):
	devDataArray[i,0] = normalize(devDataList[i][0], gradeMax, gradeMin)
	devDataArray[i,1] = normalize(devDataList[i][1], gradeMax, gradeMin)
	devDataArray[i,2] = normalize(devDataList[i][2], gradeMax, gradeMin)
	devDataArray[i,3] = normalize(devDataList[i][3], gradeMax, gradeMin)
	devDataArray[i,4] = normalize(devDataList[i][4], gradeMax, gradeMin)

#print(trainDataArray)
#print(trainKeyArray)
#print(devDataArray)
devWeight2Array_GD = devWeight2Array
devWeight1Array_GD = devWeight1Array
#print(devWeight1Array_GD)
#print(devWeight2Array_GD)
numIterations = 50
lossArray = np.zeros([1,numIterations])
a = 1.26819*13
#print(lossArray)
# implement gradient descent
for iteration in range(numIterations): #run 100 times
	outputD = np.zeros([numTrainData,1])
	for d in range(numTrainData):
		# calculate output of network
		currData = np.ones(1)
		#print(currData)
		#print(trainDataArray[d,:])
		currData = np.append(currData,trainDataArray[d,:])
		#print(currData)
		currData = currData[:,None]
		#print(repr(devWeight1Array_GD))
		#print(repr(currData))
		hiddenLayerInput = np.dot(devWeight1Array_GD,currData)
		#print(repr(hiddenLayerInput))
		hiddenLayerOutput = vectSigmoid(hiddenLayerInput)
		hiddenLayerOutput = np.insert(hiddenLayerOutput,0,1)
		hiddenLayerOutput = hiddenLayerOutput[:,None]
		#print(repr(hiddenLayerOutput))
		#print(repr(devWeight2Array_GD))
		finalLayerInput = np.dot(devWeight2Array_GD,hiddenLayerOutput)
		#print(finalLayerInput)
		finalLayerOutput = sigmoid(finalLayerInput)
		#print(finalLayerOutput)

		# calculate error signal of output neuron
		errorOutputNeuron = -finalLayerOutput*(1-finalLayerOutput)* \
			(trainKeyArray[d]-finalLayerOutput)
		#print(finalLayerOutput)
		#print(trainKeyArray[d])
		#print(errorOutputNeuron)

		errorHiddenLayer = np.zeros([1,4])
		for i in range(4):
			#print(i)
			if i == 0:
				continue
			else:
				currOutput = hiddenLayerOutput[i,0]
				currWeight = devWeight2Array_GD[0,i]
				errorHiddenLayer[0,i] = currOutput*(1-currOutput)*currWeight* \
					errorOutputNeuron
		#print(errorHiddenLayer)

		# update network weights
		# network weights for inputs -> hidden layer
		for i in range(3): # 3 neurons (not including bias)
			for j in range(6): # 5 inputs + 1 bias input
				#print(errorHiddenLayer[0,i+1])
				devWeight1Array_GD[i,j] = devWeight1Array_GD[i,j] - \
					stepSize*errorHiddenLayer[0,i+1]*currData[j,0]
		#print(devWeight1Array_GD)

		# network weights for hiddenlayer -> output
		for i in range(4): # 4 neurons (including bias)
			devWeight2Array_GD[0,i] = devWeight2Array_GD[0,i] - \
				stepSize*errorOutputNeuron*hiddenLayerOutput[i,0]
		#print(devWeight2Array_GD)

		# calculate final output with new weights
		hiddenLayerInput = np.dot(devWeight1Array_GD,currData)
		hiddenLayerOutput = vectSigmoid(hiddenLayerInput)
		hiddenLayerOutput = np.insert(hiddenLayerOutput,0,1)
		hiddenLayerOutput = hiddenLayerOutput[:,None]
		finalLayerInput = np.dot(devWeight2Array_GD,hiddenLayerOutput)
		finalLayerOutput = sigmoid(finalLayerInput)
		#print(devWeight1Array_GD)
		#print(devWeight2Array_GD)
		#print("New Training Data")
		outputD[d,0] = finalLayerOutput

	#print((0.71-outputD[0,0])	
	loss = np.sum(np.square(trainKeyArray-outputD))*0.5
	lossArray[0,iteration] = loss
	print(loss + a)

	#print(devWeight1Array_GD)
	#print(devWeight2Array_GD)

#print(repr(lossArray))	
print('GRADIENT DESCENT TRAINING COMPLETED!')
for i in range(15):
	print(lossArray[0,i])

print('STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.')

#print(devDataArray)
#print(devWeight1Array_GD)
#print(devWeight2Array_GD)
#print(devDataArray.shape[0])
for i in range(devDataArray.shape[0]):
	currData = np.ones(1)
	currData = np.append(currData,devDataArray[i,:])
	hiddenLayerInput = np.dot(devWeight1Array_GD,currData)
	hiddenLayerOutput = vectSigmoid(hiddenLayerInput)
	hiddenLayerOutput = np.insert(hiddenLayerOutput,0,1)
	hiddenLayerOutput = hiddenLayerOutput[:,None]
	finalLayerInput = np.dot(devWeight2Array_GD,hiddenLayerOutput)
	finalLayerOutput = sigmoid(finalLayerInput)
	print(finalLayerOutput[0,0]*100)