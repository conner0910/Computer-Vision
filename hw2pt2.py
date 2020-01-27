import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import math
from random import randint
import sys

def sobelxfun(img, rows, col):
	sobelx = [[[0 for k in range(3)] for x in range(col)] for y in range(rows)]
	for i in range(rows):
		for j in range(col):
			if i - 1 >= 0 and j - 1 >= 0 and i + 1 < rows and j + 1 < col: #reg case
				leftR = img[i][j - 1][0] * 2
				topLeftR = img[i - 1][j - 1][0] * 1
				topR = img[i - 1][j][0] * 0
				topRightR = img[i - 1][j + 1][0] * -1
				rightR = img[i][j + 1][0] * -2
				botRightR = img[i + 1][j + 1][0] * -1
				botR = img[i + 1][j][0] * 0
				botLeftR = img[i + 1][j - 1][0] * 1
				sumR = leftR + topLeftR + topR + topRightR + rightR + botRightR + botR + botLeftR
				leftG = img[i][j - 1][1] * 2
				topLeftG = img[i - 1][j - 1][0] * 1
				topG = img[i - 1][j][1] * 0
				topRightG = img[i - 1][j + 1][1] * -1
				rightG = img[i][j + 1][1] * -2
				botRightG = img[i + 1][j + 1][1] * -1
				botG = img[i + 1][j][1] * 0
				botLeftG = img[i + 1][j - 1][1] * 1
				sumG = leftG + topLeftG + topG + topRightG + rightG + botRightG + botG + botLeftG
				leftB = img[i][j - 1][2] * 2
				topLeftB = img[i - 1][j - 1][2] * 1
				topB = img[i - 1][j][2] * 0
				topRightB = img[i - 1][j + 1][2] * -1
				rightB = img[i][j + 1][2] * -2
				botRightB = img[i + 1][j + 1][2] * -1
				botB = img[i + 1][j][2] * 0
				botLeftB = img[i + 1][j - 1][2] * 1
				sumB = leftB + topLeftB + topB + topRightB + rightB + botRightB + botB + botLeftB
				sobelx[i][j] = [abs(sumR), abs(sumG), abs(sumB)]
	return sobelx

def sobelyfun(img, rows, col):
	sobely = [[[0 for k in range(3)] for x in range(col)] for y in range(rows)]
	for i in range(rows):
		for j in range(col):
			if i - 1 >= 0 and j - 1 >= 0 and i + 1 < rows and j + 1 < col: #reg case
				leftR = img[i][j - 1][0] * 0
				topLeftR = img[i - 1][j - 1][0] * 1
				topR = img[i - 1][j][0] * 2
				topRightR = img[i - 1][j + 1][0] * 1
				rightR = img[i][j + 1][0] * 0
				botRightR = img[i + 1][j + 1][0] * -1
				botR = img[i + 1][j][0] * -2
				botLeftR = img[i + 1][j - 1][0] * -1
				sumR = leftR + topLeftR + topR + topRightR + rightR + botRightR + botR + botLeftR
				leftG = img[i][j - 1][1] * 0
				topLeftG = img[i - 1][j - 1][1] * 1
				topG = img[i - 1][j][1] * 2
				topRightG = img[i - 1][j + 1][1] * 1
				rightG = img[i][j + 1][1] * 0
				botRightG = img[i + 1][j + 1][1] * -1
				botG = img[i + 1][j][1] * -2
				botLeftG = img[i + 1][j - 1][1] * -1
				sumG = leftG + topLeftG + topG + topRightG + rightG + botRightG + botG + botLeftG
				leftB = img[i][j - 1][2] * 0
				topLeftB = img[i - 1][j - 1][2] * 1
				topB = img[i - 1][j][2] * 2
				topRightB = img[i - 1][j + 1][2] * 1
				rightB = img[i][j + 1][2] * 0
				botRightB = img[i + 1][j + 1][2] * -1
				botB = img[i + 1][j][2] * -2
				botLeftB = img[i + 1][j - 1][2] * -1
				sumB = leftB + topLeftB + topB + topRightB + rightB + botRightB + botB + botLeftB
				sobely[i][j] = [abs(sumR), abs(sumG), abs(sumB)]
	return sobely




def main():
	img1 = cv2.imread("wt_slic.png", cv2.IMREAD_COLOR)
	rows = len(img1)
	cols = len(img1[0])
	gradient = [[[0 for k in range(3)] for x in range(cols)] for y in range(rows)]
	gradientx = sobelxfun(img1, rows, cols)
	gradienty = sobelyfun(img1, rows, cols)
	for i in range(rows):
		for j in range(cols):
			gradient[i][j][0] = math.sqrt(gradientx[i][j][0] ** 2 + gradienty[i][j][0] ** 2)
			gradient[i][j][1] = math.sqrt(gradientx[i][j][1] ** 2 + gradienty[i][j][1] ** 2)
			gradient[i][j][2] = math.sqrt(gradientx[i][j][2] ** 2 + gradienty[i][j][2] ** 2) 

	combinedGrad = [[0 for x in range(cols)] for y in range(rows)]
	for i in range(rows):
		for j in range(cols):
			combinedGrad[i][j] = math.sqrt(gradient[i][j][0] ** 2 + gradient[i][j][1] ** 2 + gradient[i][j][2] ** 2)

	centroids = [[0 for k in range(5)] for x in range(150)]
	oldcentroids = [[0 for k in range(5)] for x in range(150)]
	count = 0
	membership = [[0 for k in range(cols)] for j in range(rows)]
	for i in range(24, rows, 50):
		for j in range(24, cols, 50):
			minGrad = min(combinedGrad[i][j], combinedGrad[i][j - 1], combinedGrad[i - 1][j - 1], combinedGrad[i - 1][j], combinedGrad[i - 1][j + 1], combinedGrad[i][j + 1], combinedGrad[i + 1][j + 1], combinedGrad[i + 1][j], combinedGrad[i + 1][j - 1])
			if minGrad == combinedGrad[i][j]:
				centroids[count] = [i, j, img1[i][j][0], img1[i][j][1], img1[i][j][2]]
			elif minGrad == combinedGrad[i][j - 1]:
				centroids[count] = [i, j - 1, img1[i][j - 1][0], img1[i][j - 1][1], img1[i][j - 1][2]]
			elif minGrad == combinedGrad[i - 1][j - 1]:
				centroids[count] = [i - 1, j - 1, img1[i - 1][j - 1][0], img1[i - 1][j - 1][1], img1[i - 1][j - 1][2]]
			elif minGrad == combinedGrad[i - 1][j]:
				centroids[count] = [i - 1, j, img1[i - 1][j][0], img1[i - 1][j][1], img1[i - 1][j][2]]
			elif minGrad == combinedGrad[i - 1][j + 1]:
				centroids[count] = [i - 1, j + 1, img1[i - 1][j + 1][0], img1[i - 1][j + 1][1], img1[i - 1][j + 1][2]]				
			elif minGrad == combinedGrad[i][j + 1]:
				centroids[count] = [i, j + 1, img1[i][j + 1][0], img1[i][j + 1][1], img1[i][j + 1][2]]				
			elif minGrad == combinedGrad[i + 1][j + 1]:
				centroids[count] = [i + 1, j + 1, img1[i + 1][j + 1][0], img1[i + 1][j + 1][1], img1[i + 1][j + 1][2]]				
			elif minGrad == combinedGrad[i + 1][j]:
				centroids[count] = [i + 1, j, img1[i + 1][j][0], img1[i + 1][j][1], img1[i + 1][j][2]]				
			elif minGrad == combinedGrad[i + 1][j - 1]:
				centroids[count] = [i + 1, j - 1, img1[i + 1][j - 1][0], img1[i + 1][j - 1][1], img1[i + 1][j - 1][2]]
			count += 1
	count = 0

	dist = [[[0 for x in range(150)] for y in range(cols)] for j in range(rows)]

	#gives initial membership. Get distance to all of them and get the min. use for loop and classify according to k + 1
	
	while(True):	
		for i in range(rows):
			for j in range(cols):
				for k in range(150):
					dist[i][j][k] = math.sqrt((((i - centroids[k][0])) ** 2) / 2 + (((j - centroids[k][1])) ** 2) / 2 + (img1[i][j][0] - centroids[k][2]) ** 2 + (img1[i][j][1] - centroids[k][3]) ** 2 + (img1[i][j][2] - centroids[k][4]) ** 2)
		print "GOT DISTANCE"
		for i in range(rows):
			for j in range(cols):
				minDist = np.min(dist[i][j])
				for k in range(150):
					if minDist == dist[i][j][k]:
						membership[i][j] = k
						break
		print "MEMBERSHIP"
		countCent = [0 for x in range(150)]
		for i in range(rows):
			for j in range(cols):
				for k in range(150):
					if membership[i][j] == k:
						centroids[k][0] += i
						centroids[k][1] += j
						centroids[k][2] += img1[i][j][0] #ignore warnings here
						centroids[k][3] += img1[i][j][1]
						centroids[k][4] += img1[i][j][2]
						countCent[k] += 1
						break
		for i in range(150):
			if countCent[i] == 0:
				countCent[i] = 1
		for k in range(150):
			centroids[k][0] /= countCent[k]
			centroids[k][1] /= countCent[k]
			centroids[k][2] /= countCent[k]
			centroids[k][3] /= countCent[k]
			centroids[k][4] /= countCent[k]
		if np.array_equal(np.array(oldcentroids), np.array(centroids)):
			break
		for k in range(150):
			oldcentroids[k][0] = centroids[k][0]
			oldcentroids[k][1] = centroids[k][1]
			oldcentroids[k][2] = centroids[k][2]
			oldcentroids[k][3] = centroids[k][3]
			oldcentroids[k][4] = centroids[k][4]

	avgRGB = [[0 for i in range(3)] for j in range(150)]
	countAvg = [0 for i in range(150)]
	for i in range(rows):
		for j in range(cols):
			for k in range(150):
				if membership[i][j] == k:
					avgRGB[k][0] += img1[i][j][0]
					avgRGB[k][1] += img1[i][j][1]
					avgRGB[k][2] += img1[i][j][2]
					countAvg[k] += 1
					break
	for i in range(150):
			if countAvg[i] == 0:
				countAvg[i] = 1
	for k in range(150):
		avgRGB[k][0] /= countAvg[k]
		avgRGB[k][1] /= countAvg[k]	
		avgRGB[k][2] /= countAvg[k]
	print "GOT AVG RGB"
	for i in range(rows):
		for j in range(cols):
			for k in range(150):
				if membership[i][j] == k:
					img1[i][j] = avgRGB[k]
					break
	print "DONE"
	for i in range(rows):
		for j in range(cols):
			if i - 1 >= 0 and j - 1 >= 0 and i + 1 < rows and j + 1 < cols:
				if membership[i][j] != membership[i][j - 1] or membership[i][j] != membership[i - 1][j - 1] or membership[i][j] != membership[i - 1][j] or membership[i][j] != membership[i - 1][j + 1] or membership[i][j] != membership[i][j + 1] or membership[i][j] != membership[i + 1][j + 1] or membership[i][j] != membership[i + 1][j] or membership[i][j] != membership[i + 1][j - 1]:
					img1[i][j][0] = 0
					img1[i][j][1] = 0
					img1[i][j][2] = 0
	cv2.imwrite("hw2pt2.png", img1)


if __name__ == '__main__':
	main()