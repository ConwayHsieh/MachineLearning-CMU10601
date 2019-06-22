# Conway Hsieh
# 10-601
# 10/02/2018

# import needed modules
import sys, csv, math, string

# import written functions
from func import H, condH, classLabel

#filename is input 1
filename = sys.argv[1]

# convert CSV contents into a list of lists
dataList = list();
with open(filename) as inputFile:
    readCSV = csv.reader(inputFile, delimiter=',')
    for row in readCSV:
    	dataList.append(row)

# first row is the name of attr
labelList = dataList[0]

# delete first row to keep only data points
del dataList[0]
#print(labelList)
#print(dataList)

# prepare object of the class
classL = classLabel(len(labelList)-1, labelList[:-1], str(
	labelList[-1]).strip())
#print(classL.numAttr)
#print(classL.attrNameList)
#print(classL.label)
#print(classL.attrList)
#print(classL.label1)
#print(classL.label2)
#print(dataList[0][classL.numAttr])

sizeData = len(dataList)
numLabel1 = 0;
for currData in dataList:
	if (currData[-1] == classL.label1):
		numLabel1 += 1

numLabel2 = sizeData - numLabel1
minLabel = min([numLabel1,numLabel2])

#print(numLabel1)
#print(numLabel2)
#print(sizeData)
print('Entropy: %f' % H(numLabel1/sizeData))
print('error: %f' % (minLabel/sizeData))