import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import math
from random import randint
import sys
import os
import glob

def histogram(img, numBins, rows, cols):
	bins = [[0 for y in range(3)] for x in range(numBins)]
	binWidth = 256 / numBins
	for i in range(rows):
		for j in range(cols):
			bins[img[i][j][0] / binWidth][0] += 1
			bins[img[i][j][1] / binWidth][1] += 1
			bins[img[i][j][2] / binWidth][2] += 1
	return bins #Returns the histogram 	



def main():
	trainHists = [0 for y in range(12)]
	testHists = [0 for y in range(12)]
	trainClass = ['' for y in range(12)]
	testClass = ['' for y in range(12)]
	trainClass[0] = 'Coast'
	trainClass[1] = 'Coast'
	trainClass[2] = 'Coast'
	trainClass[3] = 'Coast'
	trainClass[4] = 'Forest'
	trainClass[5] = 'Forest'
	trainClass[6] = 'Forest'
	trainClass[7] = 'Forest'
	trainClass[8] = 'Inside City'
	trainClass[9] = 'Inside City'
	trainClass[10] = 'Inside City'
	trainClass[11] = 'Inside City'

	#starts 8 bin 1-nearest neighbor
	a = 0
	os.chdir('ImClass')
	for filename in glob.glob('*train*'):
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		trainHists[a] = histogram(img, 8, len(img), len(img[0]))		
		a += 1
	a = 0
	for filename in glob.glob('*test*'):	
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		testHists[a] = histogram(img, 8, len(img), len(img[0]))
		a += 1

	for i in range(12): #for every test image
		teHist = testHists[i]
		min = 10000000000000000000000
		for j in range(12): #for every training img			
			trHist = trainHists[j]
			dist = 0
			for k in range(8): #for every bin
				dist += (trHist[k][0] - teHist[k][0]) ** 2
				dist += (trHist[k][1] - teHist[k][1]) ** 2
				dist += (trHist[k][2] - teHist[k][2]) ** 2
			dist = math.sqrt(dist)
			if dist < min:
				min = dist
				testClass[i] = trainClass[j]
	print "8 Bin 1-Nearest Neighbor"
	print "Train Set"
	print trainClass
	print "Test Set"
	count = 0
	for filename in glob.glob('*test*'):
		print filename + " of class " + trainClass[count] + " has been assigned to class " + testClass[count]
		count += 1
	count = 0

	#starts 4 bin 1-nearest neighbor
	a = 0
	for filename in glob.glob('*train*'):
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		trainHists[a] = histogram(img, 4, len(img), len(img[0]))		
		a += 1
	a = 0
	for filename in glob.glob('*test*'):	
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		testHists[a] = histogram(img, 4, len(img), len(img[0]))
		a += 1

	for i in range(12): #for every test image
		teHist = testHists[i]
		min = 10000000000000000000000
		for j in range(12): #for every training img			
			trHist = trainHists[j]
			dist = 0
			for k in range(4): #for every bin
				dist += (trHist[k][0] - teHist[k][0]) ** 2
				dist += (trHist[k][1] - teHist[k][1]) ** 2
				dist += (trHist[k][2] - teHist[k][2]) ** 2
			dist = math.sqrt(dist)
			if dist < min:
				min = dist
				testClass[i] = trainClass[j]
	print "4 Bin 1-Nearest Neighbor"
	print "Train Set"
	print trainClass
	print "Test Set"
	count = 0
	for filename in glob.glob('*test*'):
		print filename + " of class " + trainClass[count] + " has been assigned to class " + testClass[count]
		count += 1
	count = 0

	#starts 16 bin 1-nearest neighbor
	a = 0
	for filename in glob.glob('*train*'):
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		trainHists[a] = histogram(img, 16, len(img), len(img[0]))
		a += 1
	a = 0
	for filename in glob.glob('*test*'):	
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		testHists[a] = histogram(img, 16, len(img), len(img[0]))
		a += 1

	for i in range(12): #for every test image
		teHist = testHists[i]
		min = 10000000000000000000000
		for j in range(12): #for every training img			
			trHist = trainHists[j]
			dist = 0
			for k in range(16): #for every bin
				dist += (trHist[k][0] - teHist[k][0]) ** 2
				dist += (trHist[k][1] - teHist[k][1]) ** 2
				dist += (trHist[k][2] - teHist[k][2]) ** 2
			dist = math.sqrt(dist)
			if dist < min:
				min = dist
				testClass[i] = trainClass[j]
	print "16 Bin 1-Nearest Neighbor"
	print "Train Set"
	print trainClass
	print "Test Set"
	count = 0
	for filename in glob.glob('*test*'):
		print filename + " of class " + trainClass[count] + " has been assigned to class " + testClass[count]
		count += 1
	count = 0

	#starts 32 bin 1-nearest neighbor
	a = 0
	for filename in glob.glob('*train*'):
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		trainHists[a] = histogram(img, 32, len(img), len(img[0]))		
		a += 1
	a = 0
	for filename in glob.glob('*test*'):
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		testHists[a] = histogram(img, 32, len(img), len(img[0]))
		a += 1

	for i in range(12): #for every test image
		teHist = testHists[i]
		min = 10000000000000000000000
		for j in range(12): #for every training img			
			trHist = trainHists[j]
			dist = 0
			for k in range(32): #for every bin
				dist += (trHist[k][0] - teHist[k][0]) ** 2
				dist += (trHist[k][1] - teHist[k][1]) ** 2
				dist += (trHist[k][2] - teHist[k][2]) ** 2
			dist = math.sqrt(dist)
			if dist < min:
				min = dist
				testClass[i] = trainClass[j]
	print "32 Bin 1-Nearest Neighbor"
	print "Train Set"
	print trainClass
	print "Test Set"
	count = 0
	for filename in glob.glob('*test*'):
		print filename + " of class " + trainClass[count] + " has been assigned to class " + testClass[count]
		count += 1
	count = 0
	

	#starts 8 bin 3-nearest neighbor
	distArr = [[0 for x in range(12)] for y in range(12)] #will keep track of distances for nearest neighbors (24D distance between each test img and each training img) rows are test cols are train

	a = 0
	for filename in glob.glob('*train*'):
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		trainHists[a] = histogram(img, 8, len(img), len(img[0]))		
		a += 1
	a = 0
	for filename in glob.glob('*test*'):	
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		testHists[a] = histogram(img, 8, len(img), len(img[0]))
		a += 1

	for i in range(12): #for every test image
		teHist = testHists[i]
		min = 10000000000000000000000
		for j in range(12): #for every training img			
			trHist = trainHists[j]
			dist = 0
			for k in range(8): #for every bin
				dist += (trHist[k][0] - teHist[k][0]) ** 2
				dist += (trHist[k][1] - teHist[k][1]) ** 2
				dist += (trHist[k][2] - teHist[k][2]) ** 2
			dist = math.sqrt(dist)
			distArr[i][j] = dist
	for i in range(12):		
		dists = distArr[i]
		first = dists.index(np.amin(dists))
		dists[dists.index(np.amin(dists))] = 10000000000000000000
		second = dists.index(np.amin(dists))
		dists[dists.index(np.amin(dists))] = 10000000000000000000
		third = dists.index(np.amin(dists))
		dists[dists.index(np.amin(dists))] = 10000000000000000000
		if first >= 0 and first <= 3:
			first = 1 #class of coast
		elif first >= 4 and first <= 7:
			first = 2 #class of forest
		elif first >= 8 and first <= 11:
			first = 3 #class of insidecity
		if second >= 0 and second <= 3:
			second = 1 #class of coast
		elif second >= 4 and second <= 7:
			second = 2 #class of forest
		elif second >= 8 and second <= 11:
			second = 3 #class of insidecity
		if third >= 0 and third <= 3:
			third = 1 #class of coast
		elif third >= 4 and third <= 7:
			third = 2 #class of forest
		elif third >= 8 and third <= 11:
			third = 3 #class of insidecity


		if (first == second and first == 1) or (second == third and second == 1) or (first == third and first == 1):
			testClass[i] = "Coast"
		elif (first == second and first == 2) or (second == third and second == 2) or (first == third and first == 2):
			testClass[i] = "Forest"
		elif (first == second and first == 3) or (second == third and second == 3) or (first == third and first == 3):
			testClass[i] = "insidecity"
		else:
			testClass[i] = "Indecisive"



	print "8 Bin 3-Nearest Neighbor"
	print "Train Set"
	print trainClass
	print "Test Set"
	count = 0
	for filename in glob.glob('*test*'):
		print filename + " of class " + trainClass[count] + " has been assigned to class " + testClass[count]
		count += 1
	count = 0




	

if __name__ == '__main__':
	main()