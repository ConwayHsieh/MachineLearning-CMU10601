# Conway Hsieh
# File for functions to be called

import math, numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def normalize(value,max,min):
	if value == 'yes':
		x = 1
	elif value == 'no':
		x = 0
	else:
		x = float(value)
	return (x-min)/(max-min)