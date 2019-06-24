# Conway Hsieh
# 10-601 Naive Bayes
# 11/23/2018

# import libraries
import re, collections, sys, math, string, pprint, numpy as np

pp = pprint.PrettyPrinter(indent=4)

#filename is input 1
trainFileName = sys.argv[1]
testFileName = sys.argv[2]
q = float(sys.argv[3])

#create list to store train file names
with open(trainFileName,"r") as f:
    trainFiles = f.readlines()
with open(testFileName, "r") as f:
	testFiles = f.readlines()

# strip \n
trainFiles = [x.strip() for x in trainFiles] 
testFiles = [x.strip() for x in testFiles]

#pp.pprint(trainFiles)

libWordCounter = collections.Counter()
conWordCounter = collections.Counter()
masterWordCounter = collections.Counter()
masterWordList = list()
numLib = 0
numCon = 0
sizeLibText = 0
sizeConText = 0
for numFile in range(len(trainFiles)):
	# read in all words as lower case 
	words = re.findall(r'[A-Za-z\']+(?:\`[A-Za-z]+)?', \
		open(trainFiles[numFile], encoding="latin-1").read().lower())
	# create counter based on current doc
	wordCount = collections.Counter(words)

	masterWordCounter += wordCount
	
	# append list of all words in current document to master list
	masterWordList.append(words)

	# add counters to respective counter obj
	if re.match('^lib',trainFiles[numFile]):
		libWordCounter += wordCount
		numLib += 1
		sizeLibText += len(words)
	else:
		conWordCounter += wordCount
		numCon += 1
		sizeConText += len(words)

numTexts = numLib + numCon
pLib = numLib/numTexts
pCon = numCon/numTexts

# flatten list to 1D
masterWordList = [item for sublist in masterWordList for item in sublist]
totalWords = len(masterWordList)

# create list of unique words from all training texts
masterWordSet = list(set(masterWordList))
numUniqueWords = len(masterWordSet)

#print(totalWords)
#print(numUniqueWords)

# start testing
numCorrect = 0
#print(testFiles)
for numFile in range(len(testFiles)):
	words = re.findall(r'[A-Za-z\']+(?:\`[A-Za-z]+)?', \
		open(testFiles[numFile], encoding="latin-1").read().lower())
	#print(testFiles[numFile])
	# create counter based on current doc
	wordCount = collections.Counter(words)
	# create list of only all words in current doc

	currPLib = math.log(pLib)
	currPCon = math.log(pCon)

	#print(pLib)
	#print(currPLib)
	#print(currPCon)

	# for each word
	for word in range(len(words)):
		if masterWordCounter[words[word]] == 0:
			continue
		#print(words[word])
		numLibOcc = libWordCounter[words[word]]
		numConOcc = conWordCounter[words[word]]
		#print(numLibOcc)
		#print(numConOcc)

		try:
			currPLib += math.log((numLibOcc + q)/(sizeLibText + q*numUniqueWords))
			currPCon += math.log((numConOcc + q)/(sizeConText + q*numUniqueWords))
		except:
			pass
		#print(currPLib)
		#print(currPCon)

	#currPLib = abs(currPLib)
	#currPCon = abs(currPCon)
	#print(currPLib)
	#print(currPCon)
	if currPLib >= currPCon:
		lib = True
		print('L')
	else:
		lib = False
		print('C')

	if re.match('^lib',testFiles[numFile]):
		isLib = True
	else:
		isLib = False
	#print(isLib)

	if isLib == lib:
		numCorrect += 1

print('Accuracy: {:.04f}'.format(numCorrect/len(testFiles)))
