# Conway Hsieh
# 10-601
# Homework 1


import sys #import sys in order to take cmd line argument
import string #import str in order to determine number of lines to be read

#filename is input 1
filename = sys.argv[1]

#open file
inputFile = open(filename,"r")

#determine # of lines read
wholeString = inputFile.read()
numLines = wholeString.count("\n")

#create list to store strings
inputStrings = list()
#reopen file since it was completely read
inputFile = open(filename,"r")
for i in range(numLines): #for all newlines counted, append to list
	inputStrings.append(inputFile.readline())

#create list to store reversed strings
outputStrings = list()
for i in reversed(range(numLines)): #in reverse, append strings to list
	outputStrings.append(inputStrings[i])

#combine list into one string
outputStrings = ''.join(outputStrings)
print(outputStrings)