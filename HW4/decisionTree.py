# Conway Hsieh
# 10-601 HW4
# 10/02/2018

# import needed modules
import sys, csv, math, string, numpy as np
np.set_printoptions(threshold=np.inf)

# import written functions
from func import H, condH, MI, classLabel, Tree

#filename is input 1,2
trainFileName = sys.argv[1]
testFileName = sys.argv[2]

# convert CSV contents into a list of lists
trainDataList = list();
with open(trainFileName) as inputFile1:
    readCSV = csv.reader(inputFile1, delimiter=',')
    for row in readCSV:
    	trainDataList.append(row)

testDataList = list();
with open(testFileName) as inputFile2:
	readCSV = csv.reader(inputFile2, delimiter=',')
	for row in readCSV:
		testDataList.append(row)

# first row is the name of attr, should be same for both test and train
labelList = trainDataList[0]

# delete first row to keep only data points
del trainDataList[0]
del testDataList[0]
#print(labelList)
#print(trainDataList)

# prepare object of the class
classL = classLabel(len(labelList)-1, labelList[:-1], str(
	labelList[-1]).strip())

# prepare training data into numpy array
numTrainData = len(trainDataList)
trainDataArray = np.zeros((numTrainData,classL.numAttr))
trainResultsArray = np.zeros([numTrainData,1])
currLabelY = classL.label1
for i in range(numTrainData):
	if currLabelY == trainDataList[i][-1]:
		trainResultsArray[i] = 1
	
	for j in range(classL.numAttr):
		currAttrY = classL.attrList[j].yesLabel
		if currAttrY == trainDataList[i][j]:
			trainDataArray[i][j] = 1

# prepare test data into numpy array
numTestData = len(testDataList)
testDataArray = np.zeros((numTestData, classL.numAttr))
testResultsArray = np.zeros([numTestData,1])
for i in range(numTestData):
	if currLabelY == testDataList[i][-1]:
		testResultsArray[i] = 1

	for j in range(classL.numAttr):
		currAttrY = classL.attrList[j].yesLabel
		if currAttrY == testDataList[i][j]:
			testDataArray[i][j] = 1

#print(trainDataArray)
#print(trainResultsArray)
#print(trainDataArray.sum(axis=0))
#print(testDataArray)
#print(testResultsArray)

# start training tree
numLabelY = np.sum(trainResultsArray) # num of Yes in Label
numLabelN = numTrainData - numLabelY

trainTree = Tree(trainDataArray, trainResultsArray, classL, testDataArray, 
	testResultsArray)
# first output
#print('[%d+/%d-]' % (numLabelY,numLabelN))
trainTree.setLabelDist([numLabelY, numLabelN])

numAttrYArray = trainDataArray.sum(axis=0) # num of Yes (1's) per attribute
pAttrYArray = numAttrYArray/numTrainData # probability of Yes per attribute
pLabelY = numLabelY/numTrainData # probability of Yes in Label 

# calculate entropy for each attribute
HAttrArray = np.zeros([classL.numAttr])
for i in range(classL.numAttr):
	HAttrArray[i] = H(pAttrYArray[i])

HLabel = H(pLabelY) # calculate entropy for label

# for root node, MI(Y;X) = H(Y) - H(Y|X), X = label, Y = attributes
# calculate max MI

labelYInd = np.nonzero(trainResultsArray)[0] # indices where label is 1
labelNInd = np.nonzero(trainResultsArray==0)[0] # indices where label is 0

MIAttrArray = np.zeros([classL.numAttr])
for i in range(classL.numAttr):
	# for all label is 1, calc prob attr = 1
	a = trainDataArray[[labelYInd],[i]][0]
	totalY = sum(a)
	probY = totalY/len(labelYInd)

	#for all label is 0, calc prob attr = 1
	b = trainDataArray[[labelNInd],[i]][0]
	totalN = sum(b)
	probN = totalN/len(labelNInd)

	# calculate MI for both attributes
	MIAttrArray[i] = MI(pAttrYArray[i],pLabelY,probY,probN)

#print(MIAttrArray)
maxMI = max(MIAttrArray)
maxMI_AttrInd_root = MIAttrArray.argmax(axis=0) #maximum MI attribute - index of location
trainTree.setRootAttrIndex(maxMI_AttrInd_root)
#print(maxMI_AttrInd_root)

# follow right path: max attribute yes
# trim data so only max attr = 1 remains for all other labels
maxAttr_root_ind1 = np.nonzero(trainDataArray[:,maxMI_AttrInd_root])[0]
rightPathData = trainDataArray[maxAttr_root_ind1,:]
rightPathData = np.delete(rightPathData,maxMI_AttrInd_root,axis=1)
rightPathResults = trainResultsArray[maxAttr_root_ind1]

#print(maxAttr_root_ind1)
#print(rightPathData)
#print(rightPathData.shape[1])
#print(rightPathResults)
#print(sum(rightPathResults)[0])
#print(classL.attrList[maxMI_AttrInd_root].name)

# print root node y info
#print('%s = y: [%d+/%d-]' % (classL.attrList[maxMI_AttrInd_root].name,
#	sum(rightPathResults)[0],len(rightPathResults)-sum(rightPathResults)[0]))
trainTree.setRootAttrName(classL.attrList[maxMI_AttrInd_root].name)
trainTree.setRootYDist([sum(rightPathResults)[0],
	len(rightPathResults)-sum(rightPathResults)[0]])

# check if Root node fully sorts
if ~(sum(rightPathResults)[0] == 0 or len(rightPathResults)-sum(
	rightPathResults)[0]==0):

	#print(rightPathData)
	#print(rightPathResults)

	# if only 1 attribute left
	if rightPathData.shape[1] == 1:
		#find attribute index
		if maxMI_AttrInd_root == 0:
			maxRightAttrInd = 1
		else:
			maxRightAttrInd = 0

		# indices where max attribute of right path = 1
		maxRightYInd = np.nonzero(rightPathData)[0] 
		# print(maxRightYInd)

		trainTree.setRightChildAttrName(classL.attrList[maxRightAttrInd].name)
		if len(maxRightYInd) != 0:
		#	print('| %s = y: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
		#		sum(rightPathResults[maxRightYInd]),
		#		len(maxRightYInd)-sum(rightPathResults[maxRightYInd])))
			trainTree.setRightChildYDist([sum(rightPathResults[maxRightYInd]),
			len(maxRightYInd) - sum(rightPathResults[maxRightYInd])])

		maxRightNInd = np.nonzero(rightPathData==0)[0]
		#print(maxRightNInd)
		#print(len(maxRightNInd))
		if len(maxRightNInd) != 0:
		#	print('| %s = n: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
		#		sum(rightPathResults[maxRightNInd]),
		#		len(maxRightNInd)-sum(rightPathResults[maxRightNInd])))
			trainTree.setRightChildNDist([sum(rightPathResults[maxRightNInd]),
				len(maxRightNInd) - sum(rightPathResults[maxRightNInd])])
	else: # more than 1 attribute left
		#print("TODO")
		# calculate MI for all remaining attributes
		# we have rightPathData & rightPathResults
		# calculate entropy for each remaining attribute
		numAttrYArray_rightPath = rightPathData.sum(axis=0)
		pAttrYArray_rightPath = numAttrYArray_rightPath/len(rightPathResults)
		#print(numAttrYArray_rightPath)
		#print(pAttrYArray_rightPath)

		HAttrArray_rightPath = np.zeros([classL.numAttr-1])
		for i in range(classL.numAttr-1):
			HAttrArray_rightPath[i] = H(pAttrYArray_rightPath[i])
		#print(HAttrArray_rightPath)

		# calculate entropy for label at this node
		plabelY_rightPath = sum(rightPathResults)/len(rightPathResults)
		HLabel_rightPath = H(plabelY_rightPath) # calculate entropy for label
		#print(plabelY_rightPath)
		#print(HLabel_rightPath)

		# for root node, MI(Y;X) = H(Y) - H(Y|X), X = label, Y = attributes
		# calculate max MI

		labelYInd_rightPath = np.nonzero(
			rightPathResults)[0] # indices where label is 1 on right path
		labelNInd_rightPath	 = np.nonzero(
			rightPathResults==0)[0] # indices where label is 0 on right path
		# print(labelNInd_rightPath)
		# print(labelYInd_rightPath)
		
		MIAttrArray_rightPath = np.zeros([classL.numAttr-1])
		for i in range(classL.numAttr-1):
			# for all label is 1, calc prob attr = 1
			a = rightPathData[[labelYInd_rightPath],[i]][0]
			totalY = sum(a)
			probY = totalY/len(labelYInd_rightPath)

			#for all label is 0, calc prob attr = 1
			b = rightPathData[[labelNInd_rightPath],[i]][0]
			totalN = sum(b)
			probN = totalN/len(labelNInd_rightPath)

			# calculate MI for both attributes
			MIAttrArray_rightPath[i] = MI(pAttrYArray_rightPath[i],
				plabelY_rightPath, probY, probN)

		maxMI_rightPath = max(MIAttrArray_rightPath)
		#print(maxMI_rightPath)
		if maxMI_rightPath > 0.1:
			maxRightAttrInd = MIAttrArray_rightPath.argmax(axis=0) #maximum MI attribute - index of location
			if maxRightAttrInd >= trainTree.rootAttrIndex:
				maxRightAttrInd_universal = maxRightAttrInd + 1
			else:
				maxRightAttrInd_universal = maxRightAttrInd

			# input right child attribute index
			trainTree.setRightChildAttrInd(maxRightAttrInd_universal)

			# indices where max attribute of right path = 1
			maxRightYInd = np.nonzero(rightPathData[:,maxRightAttrInd])[0] 
			# print(maxRightYInd)

			trainTree.setRightChildAttrName(classL.attrList[maxRightAttrInd_universal].name)
			if len(maxRightYInd) != 0:
			#	print('| %s = y: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
			#		sum(rightPathResults[maxRightYInd]),
			#		len(maxRightYInd)-sum(rightPathResults[maxRightYInd])))
				trainTree.setRightChildYDist([sum(rightPathResults[maxRightYInd]),
				len(maxRightYInd) - sum(rightPathResults[maxRightYInd])])

			maxRightNInd = np.nonzero(rightPathData[:,maxRightAttrInd]==0)[0]
			#print(maxRightNInd)
			#print(len(maxRightNInd))
			if len(maxRightNInd) != 0:
			#	print('| %s = n: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
			#		sum(rightPathResults[maxRightNInd]),
			#		len(maxRightNInd)-sum(rightPathResults[maxRightNInd])))
				trainTree.setRightChildNDist([sum(rightPathResults[maxRightNInd]),
					len(maxRightNInd) - sum(rightPathResults[maxRightNInd])])


# follow left path: max attribute no
# trim data so only max attr = 0 remains for all other labels
maxAttr_root_ind0 = np.nonzero(trainDataArray[:,maxMI_AttrInd_root]==0)[0]
leftPathData = trainDataArray[maxAttr_root_ind0,:]
#print(leftPathData)
leftPathData = np.delete(leftPathData,maxMI_AttrInd_root,axis=1)
#print(leftPathData)
leftPathResults = trainResultsArray[maxAttr_root_ind0]
#print(leftPathResults)

# print root node n info
#print('%s = n: [%d+/%d-]' % (classL.attrList[maxMI_AttrInd_root].name,
#	sum(leftPathResults)[0],len(leftPathResults)-sum(leftPathResults)[0]))
trainTree.setRootNDist([sum(leftPathResults)[0],
	len(leftPathResults) - sum(leftPathResults)[0]])

# check if Root node fully sorts
if ~(sum(leftPathResults)[0] == 0 or len(leftPathResults)-sum(
	leftPathResults)[0]==0):

	# if only 1 attribute left
	if leftPathData.shape[1] == 1:
		#find attribute index
		if maxMI_AttrInd_root == 0:
			maxLeftAttrInd = 1
		else:
			maxLeftAttrInd = 0

		maxLeftYInd = np.nonzero(leftPathData)[0]
		#print(maxRightYInd)

		trainTree.setLeftChildAttrName(classL.attrList[maxLeftAttrInd].name)
		if len(maxLeftYInd) != 0:
		#	print('| %s = y: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
		#		sum(leftPathResults[maxLeftYInd]),
		#		len(maxLeftYInd)-sum(leftPathResults[maxLeftYInd])))
			trainTree.setLeftChildYDist([sum(leftPathResults[maxLeftYInd]),
				len(maxLeftYInd) - sum(leftPathResults[maxLeftYInd])])	

		maxLeftNInd = np.nonzero(leftPathData==0)[0]
		#print(maxLefttNInd)
		if len(maxLeftNInd) != 0:
			#print('| %s = n: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
			#	sum(leftPathResults[maxLeftNInd]),
			#	len(maxLeftNInd)-sum(leftPathResults[maxLeftNInd])))
			trainTree.setLeftChildNDist([sum(leftPathResults[maxLeftNInd]),
				len(maxLeftNInd) - sum(leftPathResults[maxLeftNInd])])
	else: # more than 1 attribute left
		#print("TODO")
		# calculate MI for all remaining attributes
		# we have leftPathData & leftPathResults
		# calculate entropy for each remaining attribute
		numAttrYArray_leftPath = leftPathData.sum(axis=0)
		pAttrYArray_leftPath = numAttrYArray_leftPath/len(leftPathResults)
		#print(numAttrYArray_leftPath)
		#print(pAttrYArray_leftPath)

		HAttrArray_leftPath = np.zeros([classL.numAttr-1])
		for i in range(classL.numAttr-1):
			HAttrArray_leftPath[i] = H(pAttrYArray_leftPath[i])
		#print(HAttrArray_leftPath)

		# calculate entropy for label at this node
		plabelY_leftPath = sum(leftPathResults)/len(leftPathResults)
		HLabel_leftPath = H(plabelY_leftPath) # calculate entropy for label
		#print(plabelY_leftPath)
		#print(HLabel_leftPath)

		# for root node, MI(Y;X) = H(Y) - H(Y|X), X = label, Y = attributes
		# calculate max MI

		labelYInd_leftPath = np.nonzero(
			leftPathResults)[0] # indices where label is 1 on left path
		labelNInd_leftPath	 = np.nonzero(
			leftPathResults==0)[0] # indices where label is 0 on left path
		# print(labelNInd_leftPath)
		# print(labelYInd_leftPath)
		
		MIAttrArray_leftPath = np.zeros([classL.numAttr-1])
		for i in range(classL.numAttr-1):
			# for all label is 1, calc prob attr = 1
			a = leftPathData[[labelYInd_leftPath],[i]][0]
			totalY = sum(a)
			probY = totalY/len(labelYInd_leftPath)

			#for all label is 0, calc prob attr = 1
			b = leftPathData[[labelNInd_leftPath],[i]][0]
			totalN = sum(b)
			probN = totalN/len(labelNInd_leftPath)

			# calculate MI for both attributes
			MIAttrArray_leftPath[i] = MI(pAttrYArray_leftPath[i],
				plabelY_leftPath, probY, probN)

		maxMI_leftPath = max(MIAttrArray_leftPath)
		#print(maxMI_leftPath)
		if maxMI_leftPath > 0.1:
			maxLeftAttrInd = MIAttrArray_leftPath.argmax(axis=0) #maximum MI attribute - index of location
			if maxLeftAttrInd >= trainTree.rootAttrIndex:
				maxLeftAttrInd_universal = maxLeftAttrInd + 1
			else:
				maxLeftAttrInd_universal = maxLeftAttrInd

			# input left child attribute index
			trainTree.setLeftChildAttrInd(maxLeftAttrInd_universal)

			# indices where max attribute of left path = 1
			maxLeftYInd = np.nonzero(leftPathData[:,maxLeftAttrInd])[0] 
			# print(maxRightYInd)

			trainTree.setLeftChildAttrName(classL.attrList[maxLeftAttrInd_universal].name)
			if len(maxLeftYInd) != 0:
			#	print('| %s = y: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
			#		sum(rightPathResults[maxRightYInd]),
			#		len(maxRightYInd)-sum(rightPathResults[maxRightYInd])))
				trainTree.setLeftChildYDist([sum(leftPathResults[maxLeftYInd]),
				len(maxLeftYInd) - sum(leftPathResults[maxLeftYInd])])

			maxLeftNInd = np.nonzero(leftPathData[:,maxLeftAttrInd]==0)[0]
			#print(maxRightNInd)
			#print(len(maxRightNInd))
			if len(maxLeftNInd) != 0:
			#	print('| %s = n: [%d+/%d-]' % (classL.attrList[maxRightAttrInd].name,
			#		sum(rightPathResults[maxRightNInd]),
			#		len(maxRightNInd)-sum(rightPathResults[maxRightNInd])))
				trainTree.setLeftChildNDist([sum(leftPathResults[maxLeftNInd]),
					len(maxLeftNInd) - sum(leftPathResults[maxLeftNInd])])


trainTree.printTree()