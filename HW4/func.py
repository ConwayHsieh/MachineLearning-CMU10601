# Conway Hsieh
# File for functions to be called

import math, numpy as np

def H(p):
	# calculate entropy
	# p is probability
	#print(p)
	a = p
	b = 1-p
	#print(a)
	#print(b)
	if a == 0 or b == 0:
		#print("is zero")
		return 0
	else:
		#print("is not zero")		
		return (a*math.log((1/a),2) + b*math.log((1/b),2))

def condH(pX0, pA0, pB0):
	# calculate conditional entropy for binary random vars
	return pX0*H(pA0) + (1-pX0)*H(pB0)
	
def MI(pY0, pX0, pA0, pB0):
	# calculate mutual information I(Y;X) = H(Y) - H(Y|X)
	return H(pY0) - condH(pX0, pA0, pB0)

class classLabel:
	def __init__(self, numAttr, attrNameList, label):
		self.numAttr = numAttr
		self.attrNameList = attrNameList
		self.label = label
		self.attrList = list()

		if label == 'Party':
			self.label1 = 'democrat'
			self.label2 = 'republican'
			if numAttr == 2:
				self.attrList.append(attribute('Anti_satellite_test_ban','y','n'))
				self.attrList.append(attribute('Export_south_africa','y','n'))

			else:
				self.attrList.append(attribute('Anti_satellite_test_ban','y','n'))
				self.attrList.append(attribute('Aid_to_nicaraguan_contras','y','n'))
				self.attrList.append(attribute('Mx_missile','y','n'))
				self.attrList.append(attribute('Immigration','y','n'))
				self.attrList.append(attribute('Superfund_right_to_sue','y','n'))
				self.attrList.append(attribute('Duty_free_exports','y','n'))
				self.attrList.append(attribute('Export_south_africa','y','n'))

		elif label == 'grade':
			self.attrList.append(attribute('M1','A','notA'))
			self.attrList.append(attribute('M2','A','notA'))
			self.attrList.append(attribute('M3','A','notA'))
			self.attrList.append(attribute('M4','A','notA'))
			self.attrList.append(attribute('M5','A','notA'))
			self.attrList.append(attribute('P1','A','notA'))
			self.attrList.append(attribute('P2','A','notA'))
			self.attrList.append(attribute('P3','A','notA'))
			self.attrList.append(attribute('P4','A','notA'))
			self.attrList.append(attribute('F','A','notA'))

			self.label1 = 'A'
			self.label2 = 'notA'

		elif label == 'hit':
			self.attrList.append(attribute('year','before1950','after1950'))
			self.attrList.append(attribute('solo','yes','no'))
			self.attrList.append(attribute('vocal','yes','no'))
			self.attrList.append(attribute('length','morethan3min','lessthan3min'))
			self.attrList.append(attribute('original','yes','no'))
			self.attrList.append(attribute('tempo','fast','slow'))
			self.attrList.append(attribute('folk','yes','no'))
			self.attrList.append(attribute('classical','yes','no'))
			self.attrList.append(attribute('rhythm','yes','no'))
			self.attrList.append(attribute('jazz','yes','no'))
			self.attrList.append(attribute('rock','yes','no'))

			self.label1 = 'yes'
			self.label2 = 'no'

		elif label == 'class':
			self.attrList.append(attribute(self.attrNameList[0],'expensive','cheap'))
			self.attrList.append(attribute(self.attrNameList[1],'high','low'))
			self.attrList.append(attribute(self.attrNameList[2],'Two','MoreThanTwo'))
			self.attrList.append(attribute(self.attrNameList[3],'morethan3min','lessthan3min'))
			self.attrList.append(attribute(self.attrNameList[4],'Two','MoreThanTwo'))
			self.attrList.append(attribute(self.attrNameList[5],'large','small'))
			self.attrList.append(attribute(self.attrNameList[6],'high','low'))

			self.label1 = 'yes'
			self.label2 = 'no'

class attribute:
	def __init__(self, name, yesLabel, noLabel):
		self.name = name
		self.yesLabel = yesLabel
		self.noLabel = noLabel

class Tree:
	def __init__(self, trainDataArray, trainResultsArray, classL,
		testDataArray,testResultsArray):
		self.trainDataArray = trainDataArray
		self.trainResultsArray = trainResultsArray
		self.classL = classL
		self.testDataArray = testDataArray
		self.testResultsArray = testResultsArray
		self.numTestData = len(testResultsArray)

		self.labelDist = list()
	
		self.rootAttrName = list()
		self.rootAttrIndex = -1
		self.rootYDist = list()

		self.rightChildAttrName = list()
		self.rightChildAttrIndex = -1
		self.rightChildYDist = list()
		self.rightChildNDist = list()

		self.rootNDist = list()
		self.leftChildAttrName = list()
		self.leftChildAttrIndex = -1
		self.leftChildYDist = list()
		self.leftChildNDist = list()

	def setLabelDist(self,x):
		self.labelDist = x

	def setRootAttrName(self,x):
		self.rootAttrName = x
		
	def setRootYDist(self,x):
		self.rootYDist = x
	
	def setRightChildAttrName(self,x):
		self.rightChildAttrName = x
		
	def setRightChildYDist(self,x):
		self.rightChildYDist = x
		
	def setRightChildNDist(self,x):
		self.rightChildNDist = x
		
	def setRootNDist(self,x):
		self.rootNDist = x
		
	def setLeftChildAttrName(self,x):
		self.leftChildAttrName = x
		
	def setLeftChildYDist(self,x):
		self.leftChildYDist = x

	def setLeftChildNDist(self,x):
		self.leftChildNDist = x

	def setRootAttrIndex(self,x):
		self.rootAttrIndex = x

	def setRightChildAttrInd(self,x):
		self.rightChildAttrIndex = x

	def setLeftChildAttrInd(self,x):
		self.leftChildAttrIndex = x

	def printErrors(self, sizeData):
		# training error
		# path1
		# if not empty right path
		if (len(self.rightChildYDist) != 0 or len(self.rightChildNDist) != 0):
			a = min(self.rightChildYDist)
			b = min(self.rightChildNDist)
			c = a + b
			rightEmpty = False
			if np.argmax(self.rightChildYDist) == 0:
				rightChildY = '+'
			else:
				rightChildY = '-'
			if np.argmax(self.rightChildNDist) == 0:
				rightChildN = '+'
			else:
				rightChildN = '-'
		else: #empty right path, so just use first node Y
			c = min(self.rootYDist)
			rightEmpty = True
			if np.argmax(self.rootYDist) == 0:
				rightRootY = '+'
			else:
				rightRootY = '-'

		# path 2
		# not empty left path
		if (len(self.leftChildYDist) != 0 or len(self.leftChildNDist) != 0):
			d = min(self.leftChildYDist)
			e = min(self.leftChildNDist)
			f = d + e
			leftEmpty = False
			if np.argmax(self.leftChildYDist) == 0:
				leftChildY = '+'
			else:
				leftChildY = '-'
			if np.argmax(self.leftChildNDist) == 0:
				leftChildN = '+'
			else:
				leftChildN = '-'
		else:
			f = min(self.rootNDist)
			leftEmpty = True
			if np.argmax(self.rootNDist) == 0:
				leftRootY = '+'
			else:
				leftRootY = '-'

		print('error(train): %f' % ((c+f)/sizeData))

		# testing error
		# we have testDataArray, testResultsArray
		# we need columns with root node, right child, left child, leaves

		testWrong = 0;
		# 1 - RIGHT SIDE OF TREE
		if rightEmpty:
			# loop through all data points
			for i in range(self.numTestData):
				# if data point is 1 for root attribute
				if self.testDataArray[i][self.rootAttrIndex] == 1:
					# Check results classification
					# root attr class +, 0 is then wrong
					if rightRootY == '+' and self.testResultsArray[i] == 0:
						#print('A')
						testWrong += 1
					#root attr class -, then 1 is wrong
					elif rightRootY == '-' and self.testResultsArray[i] == 1:
						testWrong += 1
						#print('B')
		else: #right not empty, 1 -> 1/0 | Look at right childs
			# loop through all data points
			for i in range(self.numTestData):
				# if data point is 1 for right child attribute AND root attr = 1
				if self.testDataArray[i][self.rightChildAttrIndex
				] == 1 and self.testDataArray[i][self.rootAttrIndex] == 1:
					# Check results classification
					# right child attr class +, 0 result is wrong
					if rightChildY == '+' and self.testResultsArray[i] == 0:
						testWrong += 1
						#print('C')
					# right child attr class -, 1 result is wrong
					elif rightChildY == '-' and self.testResultsArray[i] == 1:
						testWrong += 1
						#print('D')
				# data point is 0 for right attribute AND root attr = 1
				elif self.testDataArray[i][self.rightChildAttrIndex
					] == 0 and self.testDataArray[i][self.rootAttrIndex] == 1:
					# check results classification
					# right child attr +, 0 result is wrong
					if rightChildN == '+' and self.testResultsArray[i] == 0:
						testWrong += 1
						#print('E')
					# right child attr class -, 1 result is wrong
					elif rightChildN == '-' and self.testResultsArray[i] == 1:
						testWrong += 1
						#print('F')

 
		# 0 - LEFT SIDE OF TREE 
		if leftEmpty:
			# loop through all data points
			for i in range(self.numTestData):
				# if data point is 0 (LEFT) for root attribute
				if self.testDataArray[i][self.rootAttrIndex] == 0:
					# Check results classification
					# root attr class +, 0 is then wrong
					if leftRootY == '+' and self.testResultsArray[i] == 0:
						testWrong += 1
						#print('G')
					#root attr class -, then 1 is wrong
					elif leftRootY == '-' and self.testResultsArray[i] == 1:
						testWrong += 1
						#print('H')
		else: #left not empty, 0 -> 1/0 | Look at left childs
			# loop through all data points
			for i in range(self.numTestData):
				# if data point is 1 for left child attribute AND root attr = 0
				if self.testDataArray[i][self.leftChildAttrIndex
					] == 1 and self.testDataArray[i][self.rootAttrIndex] == 0:
					# Check results classification
					# left child attr class +, 0 result is wrong
					if leftChildY == '+' and self.testResultsArray[i] == 0:
						testWrong += 1
						#print('I')
					# left child attr class -, 1 result is wrong
					elif leftChildY == '-' and self.testResultsArray[i] == 1:
						testWrong += 1
						#print('J')
				# data point is 0 for left child attribute AND root attr = 0
				elif self.testDataArray[i][self.leftChildAttrIndex
					] == 0 and self.testDataArray[i][self.rootAttrIndex] == 0:
					# check results classification
					# left child attr +, 0 result is wrong
					if leftChildN == '+' and self.testResultsArray[i] == 0:
						testWrong += 1
						#print('K')
					# left child attr class -, 1 result is wrong
					elif leftChildN == '-' and self.testResultsArray[i] == 1:
						testWrong += 1
						#print('L')

		#print(testWrong)
		print('error(test): %f' % (testWrong/self.numTestData))

			
		
	def printTree(self):
		print('[%d+/%d-]' % (self.labelDist[0], self.labelDist[1]))
		print('%s = %s: [%d+/%d-]' % (self.rootAttrName, 
			self.classL.attrList[self.rootAttrIndex].yesLabel,
			self.rootYDist[0], self.rootYDist[1]))
		if ~(self.rootYDist[0] == 0 or self.rootYDist[1] == 0):
			if (len(self.rightChildYDist) != 0 or len(self.rightChildNDist) != 0):
				#print(len(self.rightChildYDist))
				print('| %s = %s: [%d+/%d-]' % (self.rightChildAttrName, 
					self.classL.attrList[self.rightChildAttrIndex].yesLabel, 
					self.rightChildYDist[0], self.rightChildYDist[1]))
				print('| %s = %s: [%d+/%d-]' % (self.rightChildAttrName, 
					self.classL.attrList[self.rightChildAttrIndex].noLabel, 
					self.rightChildNDist[0], self.rightChildNDist[1]))

		print('%s = %s: [%d+/%d-]' % (self.rootAttrName, 
			self.classL.attrList[self.rootAttrIndex].noLabel, 
			self.rootNDist[0], self.rootNDist[1]))
		if ~(self.rootNDist[0] == 0 or self.rootNDist[1] == 0):
			if len(self.leftChildYDist) != 0 or len(self.leftChildNDist) != 0:
				print('| %s = %s: [%d+/%d-]' % (self.leftChildAttrName, 
					self.classL.attrList[self.leftChildAttrIndex].yesLabel, 
					self.leftChildYDist[0], self.leftChildYDist[1]))
				print('| %s = %s: [%d+/%d-]' % (self.leftChildAttrName, 
					self.classL.attrList[self.leftChildAttrIndex].noLabel,
					self.leftChildNDist[0], self.leftChildNDist[1]))

		self.printErrors(len(self.trainResultsArray))
			