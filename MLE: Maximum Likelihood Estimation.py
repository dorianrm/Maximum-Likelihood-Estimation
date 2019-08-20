import csv
import pandas as pd
import numpy as np
import math

##### File Names #####
trainFile = "FILE NAME HERE.txt"
testFile = "FILE NAME HERE.txt"

##### Global vars #####
map = dict()
y0prob = 0.0
y1prob = 0.0

##### MLE #####

#Creates a map of values based on previous data and calculates respective probabilities
def trainFunction():
	with open(trainFile) as train, open(testFile) as test:

		##### train #####
		y0 = 0.0
		y1 = 0.0
		counter = 0
		yindex = 2  #Index of y/posterior in lists that are the keys in the map
		ycol = train.shape[1]-1  #col of label/outcome val
		key = []
		for row in train:
			if row[ycol] == 0:
				y0 += 1
			else:
				y1 += 1
			for col in row:
				if counter != (ycol): 					 #counter = x_i/col
					key.extend((counter,col,row[ycol]))  #create key
					key = tuple(key)
					if key in map:           			 #1: key in map -> add 1 to counter
						count = map[key]
						count += 1 
						map[key] = count
					else:								 #2: add key to map, counter=1
						map[key] = 1
					key = []
					counter += 1
			counter = 0

		global y0prob
		global y1prob
		y0prob = y0/train.shape[0]  #y=0 probability
		y1prob = y1/train.shape[0]	#y=1 probability

		for key in map:				#key = (x_i, val of x_i, yval) , key maps to val of P(x_i|Y)
			prob = map[key]
			if key[yindex] == 0:    #Make probabilities for P(x_i|Y)
				prob /= y0
				map[key] = prob
			else:
				prob /= y1
				map[key] = prob


#Takes values from map created in training and uses a Naiive Bayes formula to predict the corresponding label/outcome
def testFunction():
	with open(trainFile) as train, open(testFile) as test:

		##### test ####
		test0 = 0
		test1 = 1
		counter = 0.0
		p0 = y0prob
		p1 = y1prob
		ycol = train.shape[1]-1
		success = 0.0
		for row in test:
			for col in row:
				if counter != (ycol):
					if (counter, col, test0) in map:       #y=0 probabilities
						value = map[(counter, col, test0)]
						p0 *= value                        #overall prob for 0
					else: 
						p0 *= 0
					if (counter, col, test1) in map: 	   #y=1 probabilities
						value = map[(counter, col, test1)]
						p1 *= value                        #overall prob for 1
					else:
						p1 *= 0
					counter += 1
			if p0 > p1:
				if row[ycol] == 0:
					success += 1
			elif p1 > p0:
				if row[ycol] == 1:
					success += 1
			p0 = y0prob #reset vals
			p1 = y1prob
			counter = 0
		print(success/test.shape[0])

trainFunction()
testFunction()

	