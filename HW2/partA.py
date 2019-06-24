# Conway Hsieh
# 10-601
# Homework 2 - Part A
# 09/13/2018

# import needed libraries
import sys
import string

# Hard Code Answers 1-5
# 1
print(512)
#2
print(155)
# 3
print(19684)
#4
print(59050)
#5
print(26245)

# Part 6 Find-S Algorithm

# open training file
trainFile = open("9Cat-Train.labeled")

#determine # of lines read
wholeString = trainFile.read()
numTrain = wholeString.count("\n")

#create list to store strings
trainStrings = list()
#reopen file since it was completely read
trainFile = open("9Cat-Train.labeled","r")
for i in range(numTrain): #for all newlines counted, append to list
	trainStrings.append(trainFile.readline())

# first hypothesis is null, 2nd hypothesis is first data point
numCats = 9

# create function to extract the data inputs as a list
def parseData(dataPoint):
	# define output list
	outputList = list()

	#takes in a single string
	numCats = 9
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

#print(parseData(trainStrings[1]))
#print(parseSuccess(trainStrings[1]))

# prep output file
outputFile = open("partA6.txt", "w")
tab = "\t"

# run the hypothesis through all examples
firstPositiveData = True
questionString = str('?')
hypothesisList = list()
for i in range(numTrain):
	if parseSurvive(trainStrings[i]): #if not survive, pass 
		if firstPositiveData: #if
			hypothesisList = parseData(trainStrings[i])
			firstPositiveData = False
		else:
			currList = parseData(trainStrings[i])
			for cat in range(numCats):
				hypVal = hypothesisList[cat]
				if hypVal == questionString:
					continue
				else:
					currVal = currList[cat]
					if hypVal != currVal:
						hypothesisList[cat] = questionString
	#print(hypothesisList)
	if int(i+1)%20 == 0: #print every 20th hypothesis
		#print(tab.join(hypothesisList))
		print(tab.join(hypothesisList), file = outputFile)

# Part 7
finalHypothesisList = hypothesisList

#print(finalHypothesisList)
devFile = open("9Cat-Dev.labeled","r")

#determine # of lines read
wholeString = devFile.read()
numDev = wholeString.count("\n")

#create list to store strings
devStrings = list()
#reopen file since it was completely read
devFile = open("9Cat-Dev.labeled","r")
for i in range(numDev): #for all newlines counted, append to list
	devStrings.append(devFile.readline())

classIncorrect = 0
for i in range(numDev):
	currData = devStrings[i]
	currList = parseData(currData)
	currAge = currList[0]
	currGender = currList[5]

	#print(currAge + currGender + str(parseSurvive(devStrings[i])))

	if ((str(currGender) == str('Female')) & (str(currAge) == str('Young'))):
		# data point is young female
		if not parseSurvive(currData): # if not survive
				classIncorrect += 1 #misclass
	else: #data point not young female
		if parseSurvive(currData): #if any non young female survives
			classIncorrect += 1 #missclass

#print(classIncorrect)
misClass = classIncorrect/numDev
print(misClass)


# Part 8

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

# for all data points from input file, check classification and output "yes/no"
for i in range(numInput):
	currData = inputStrings[i]
	currList = parseData(currData)
	currAge = currList[0]
	currGender = currList[5]

	if ((str(currGender) == str('Female')) & (str(currAge) == str('Young'))):
		# data point is young female
		print("Yes")
	else: #data point not young female
		print("No")
		