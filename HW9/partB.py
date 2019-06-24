# Conway Hsieh
# 10-601
# Homework 2 - Part B
# 09/13/2018

# import needed libraries
import sys
import string
import itertools
import numpy as np

# Hard Code Answers 1-2
# 1
print(16)
# 2
print(65536)

# Part 3 - List then Eliminate
trainFile = open("4Cat-Train.labeled","r")
wholeString = trainFile.read()
numTrain = wholeString.count("\n")

#create list to store strings
trainStrings = list()
#reopen file since it was completely read
trainFile = open("4Cat-Train.labeled","r")
for i in range(numTrain): #for all newlines counted, append to list
	trainStrings.append(trainFile.readline())

# generate bit vector to represent hypothesis space
hypothesisSpace = list(itertools.product([0,1], repeat=16))
numHypothesis = len(hypothesisSpace) # 65536

# generate input space list
inputList = list(itertools.product([0,1],repeat=4))
#print(inputList)
#print(len(inputList))

# create function to extract the data inputs as a list
def parseData(dataPoint):
	# define output list
	outputList = list()

	#takes in a single string
	numCats = 4
	splitData = dataPoint.split("\t")

	#for all categories, split by space, and take second 
	for i in range(numCats):
		catData = splitData[i].split(" ")
		outputList.append(catData[1])
	return outputList

def parseSurvive(dataPoint):
	splitData = dataPoint.split("\t")
	yesString = str('Yes')
	surviveString = str(splitData[-1].split(" ")[1].split("\n")[0])
	return surviveString == yesString

def binaryData(dataList):
	currAge = dataList[0]
	currClass = dataList[1]
	currEmb = dataList[2]
	currSex = dataList[3]

	compList = list([0,0,0,0])

	if currAge == 'Young':
		compList[0] = 1
	if currClass == '1':
		compList[1] = 1
	if currEmb == 'Southampton':
		compList[2] = 1
	if currSex == 'Male':
		compList[3] = 1

	return tuple(compList)

#for all training data, compare to h in VS
for i in range(numTrain):
	# draw out current data into list
	currList = parseData(trainStrings[i])
	#print(currList)
	
	# reconstruct list into binary vector
	compList = binaryData(currList)
	#print(compList)
	#print(inputList)

	# determine index of this input in inputList
	currInputIndex = inputList.index(compList)

	# determine if input data survives
	currSurvive = parseSurvive(trainStrings[i])

	#determine current size of VS
	numHypothesis = len(hypothesisSpace)

	removeHyp = list()
	# for all remaining hypotheses
	for j in range(numHypothesis):
		# take out current hypothesis
		currHyp = hypothesisSpace[j]

		if currSurvive: #if survived
			if currHyp[currInputIndex] != 1: # hypothesis did not survive
				removeHyp.append(currHyp)
		else: #if died
			if currHyp[currInputIndex] != 0: #hypothesis says survive
				removeHyp.append(currHyp)

	for j in range(len(removeHyp)):
		hypothesisSpace.remove(removeHyp[j])

print(len(hypothesisSpace))

# Part 4
#filename is input 1
filename = sys.argv[1]

#open file
inputFile = open(filename,"r")

#determine # of lines read
wholeString = inputFile.read()
numInput = wholeString.count("\n")

#create list to store strings
inputStrings = list()
#reopen file since it was completely read
inputFile = open(filename,"r")
for i in range(numInput): #for all newlines counted, append to list
	inputStrings.append(inputFile.readline())

for i in range(numInput):
	currList = parseData(inputStrings[i])
	compList = binaryData(currList)
	# determine index of this input in inputList
	currInputIndex = inputList.index(compList)
	#print(compList)
	#print(currInputIndex)

	yesCount = 0
	noCount = 0

	for j in range(len(hypothesisSpace)):
		currHyp = hypothesisSpace[j]
		#print(currHyp)
		if currHyp[currInputIndex] == 1:
			yesCount += 1
		else:
			noCount += 1

	print("%d %d" %(yesCount,noCount))

