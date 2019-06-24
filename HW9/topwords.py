# Conway Hsieh
# 10-601 Naive Bayes
# 11/23/2018

# import libraries
import re, collections, sys, math, string, pprint, numpy as np

pp = pprint.PrettyPrinter(indent=4)

#filename is input 1
trainFileName = sys.argv[1]
n = 20

#create list to store train file names
with open(trainFileName,"r") as f:
    trainFiles = f.readlines()

# strip \n
trainFiles = [x.strip() for x in trainFiles] 

#pp.pprint(trainFiles)

libWordCounter = collections.Counter()
conWordCounter = collections.Counter()
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
masterWordSet = sorted(list(set(masterWordList)))
numUniqueWords = len(masterWordSet)

#print(totalWords)
#print(numUniqueWords)

pLib = np.zeros((1,numUniqueWords))[0]
pCon = np.zeros((1,numUniqueWords))[0]
#pp.pprint(pLib)
for word in range(numUniqueWords):
	currWord = masterWordSet[word]
	#print(currWord)
	numLibOcc = libWordCounter[masterWordSet[word]]
	numConOcc = conWordCounter[masterWordSet[word]]
	#print(numLibOcc)

	pLib[word] = np.log((numLibOcc + 1)/(sizeLibText + numUniqueWords))
	pCon[word] = np.log((numConOcc + 1)/(sizeConText + numUniqueWords))


#pp.pprint(pLib)

libTopInd = np.argsort(-pLib)[:n]
conTopInd = np.argsort(-pCon)[:n]


for i in range(n):
	print( masterWordSet[libTopInd[i]],'{:.04f}'.format(math.exp(pLib[libTopInd[i]])))

print('')

for i in range(n):
	print( masterWordSet[conTopInd[i]],'{:.04f}'.format(math.exp(pCon[conTopInd[i]])))