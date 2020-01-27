import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import math
from random import randint
import sys

def main():
	img = cv2.imread("white-tower.png", cv2.IMREAD_COLOR)
	rows = len(img)
	cols = len(img[0])

	rand1 = img[randint(0, rows)][randint(0, cols)]
	rand2 = img[randint(0, rows)][randint(0, cols)]
	rand3 = img[randint(0, rows)][randint(0, cols)]
	rand4 = img[randint(0, rows)][randint(0, cols)]
	rand5 = img[randint(0, rows)][randint(0, cols)]
	rand6 = img[randint(0, rows)][randint(0, cols)]
	rand7 = img[randint(0, rows)][randint(0, cols)]
	rand8 = img[randint(0, rows)][randint(0, cols)]
	rand9 = img[randint(0, rows)][randint(0, cols)]
	rand10 = img[randint(0, rows)][randint(0, cols)]

	membership = [[0 for x in range(cols)] for y in range(rows)]

	oldCentroid1 = [0, 0, 0]
	oldCentroid2 = [0, 0, 0]
	oldCentroid3 = [0, 0, 0]
	oldCentroid4 = [0, 0, 0]
	oldCentroid5 = [0, 0, 0]
	oldCentroid6 = [0, 0, 0]
	oldCentroid7 = [0, 0, 0]
	oldCentroid8 = [0, 0, 0]
	oldCentroid9 = [0, 0, 0]
	oldCentroid10 = [0, 0, 0]
	#start loop
	while(True):
		for i in range(rows):
			for j in range(cols):
					dist1 = math.sqrt((int(rand1[0]) - int(img[i][j][0])) ** 2 + (int(rand1[1]) - int(img[i][j][1])) ** 2 + (int(rand1[2]) - int(img[i][j][2])) ** 2)					
					dist2 = math.sqrt((int(rand2[0]) - int(img[i][j][0])) ** 2 + (int(rand2[1]) - int(img[i][j][1])) ** 2 + (int(rand2[2]) - int(img[i][j][2])) ** 2)
					dist3 = math.sqrt((int(rand3[0]) - int(img[i][j][0])) ** 2 + (int(rand3[1]) - int(img[i][j][1])) ** 2 + (int(rand3[2]) - int(img[i][j][2])) ** 2)
					dist4 = math.sqrt((int(rand4[0]) - int(img[i][j][0])) ** 2 + (int(rand4[1]) - int(img[i][j][1])) ** 2 + (int(rand4[2]) - int(img[i][j][2])) ** 2)
					dist5 = math.sqrt((int(rand5[0]) - int(img[i][j][0])) ** 2 + (int(rand5[1]) - int(img[i][j][1])) ** 2 + (int(rand5[2]) - int(img[i][j][2])) ** 2)
					dist6 = math.sqrt((int(rand6[0]) - int(img[i][j][0])) ** 2 + (int(rand6[1]) - int(img[i][j][1])) ** 2 + (int(rand6[2]) - int(img[i][j][2])) ** 2)
					dist7 = math.sqrt((int(rand7[0]) - int(img[i][j][0])) ** 2 + (int(rand7[1]) - int(img[i][j][1])) ** 2 + (int(rand7[2]) - int(img[i][j][2])) ** 2)
					dist8 = math.sqrt((int(rand8[0]) - int(img[i][j][0])) ** 2 + (int(rand8[1]) - int(img[i][j][1])) ** 2 + (int(rand8[2]) - int(img[i][j][2])) ** 2)
					dist9 = math.sqrt((int(rand9[0]) - int(img[i][j][0])) ** 2 + (int(rand9[1]) - int(img[i][j][1])) ** 2 + (int(rand9[2]) - int(img[i][j][2])) ** 2)
					dist10 = math.sqrt((int(rand10[0]) - int(img[i][j][0])) ** 2 + (int(rand10[1]) - int(img[i][j][1])) ** 2 + (int(rand10[2]) - int(img[i][j][2])) ** 2)
					minDist = min(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10)
					if dist1 == minDist:
						membership[i][j] = 1
					elif dist2 == minDist:
						membership[i][j] = 2
					elif dist3 == minDist:
						membership[i][j] = 3
					elif dist4 == minDist:
						membership[i][j] = 4
					elif dist5 == minDist:
						membership[i][j] = 5
					elif dist6 == minDist:
						membership[i][j] = 6
					elif dist7 == minDist:
						membership[i][j] = 7
					elif dist8 == minDist:
						membership[i][j] = 8
					elif dist9 == minDist:
						membership[i][j] = 9
					elif dist10 == minDist:
						membership[i][j] = 10

		newCentroid1 = [0, 0, 0]
		newCentroid2 = [0, 0, 0]
		newCentroid3 = [0, 0, 0]
		newCentroid4 = [0, 0, 0]
		newCentroid5 = [0, 0, 0]
		newCentroid6 = [0, 0, 0]
		newCentroid7 = [0, 0, 0]
		newCentroid8 = [0, 0, 0]
		newCentroid9 = [0, 0, 0]
		newCentroid10 = [0, 0, 0]
		count1 = 0
		count2 = 0
		count3 = 0
		count4 = 0
		count5 = 0
		count6 = 0
		count7 = 0
		count8 = 0
		count9 = 0
		count10 = 0
		for i in range(rows):
			for j in range(cols):
				if membership[i][j] == 1:
					newCentroid1[0] = newCentroid1[0] + img[i][j][0]
					newCentroid1[1] = newCentroid1[1] + img[i][j][1]
					newCentroid1[2] = newCentroid1[2] + img[i][j][2]
					count1 = count1 + 1
				elif membership[i][j] == 2:
					newCentroid2[0] = newCentroid2[0] + img[i][j][0]
					newCentroid2[1] = newCentroid2[1] + img[i][j][1]
					newCentroid2[2] = newCentroid2[2] + img[i][j][2]
					count2 = count2 + 1
				elif membership[i][j] == 3:
					newCentroid3[0] = newCentroid3[0] + img[i][j][0]
					newCentroid3[1] = newCentroid3[1] + img[i][j][1]
					newCentroid3[2] = newCentroid3[2] + img[i][j][2]
					count3 = count3 + 1
				elif membership[i][j] == 4:
					newCentroid4[0] = newCentroid4[0] + img[i][j][0]
					newCentroid4[1] = newCentroid4[1] + img[i][j][1]
					newCentroid4[2] = newCentroid4[2] + img[i][j][2]
					count4 = count4 + 1
				elif membership[i][j] == 5:
					newCentroid5[0] = newCentroid5[0] + img[i][j][0]
					newCentroid5[1] = newCentroid5[1] + img[i][j][1]
					newCentroid5[2] = newCentroid5[2] + img[i][j][2]
					count5 = count5 + 1
				elif membership[i][j] == 6:
					newCentroid6[0] = newCentroid6[0] + img[i][j][0]
					newCentroid6[1] = newCentroid6[1] + img[i][j][1]
					newCentroid6[2] = newCentroid6[2] + img[i][j][2]
					count6 = count6 + 1
				elif membership[i][j] == 7:
					newCentroid7[0] = newCentroid7[0] + img[i][j][0]
					newCentroid7[1] = newCentroid7[1] + img[i][j][1]
					newCentroid7[2] = newCentroid7[2] + img[i][j][2]
					count7 = count7 + 1
				elif membership[i][j] == 8:
					newCentroid8[0] = newCentroid8[0] + img[i][j][0]
					newCentroid8[1] = newCentroid8[1] + img[i][j][1]
					newCentroid8[2] = newCentroid8[2] + img[i][j][2]
					count8 = count8 + 1
				elif membership[i][j] == 9:
					newCentroid9[0] = newCentroid9[0] + img[i][j][0]
					newCentroid9[1] = newCentroid9[1] + img[i][j][1]
					newCentroid9[2] = newCentroid9[2] + img[i][j][2]
					count9 = count9 + 1
				elif membership[i][j] == 10:
					newCentroid10[0] = newCentroid10[0] + img[i][j][0]
					newCentroid10[1] = newCentroid10[1] + img[i][j][1]
					newCentroid10[2] = newCentroid10[2] + img[i][j][2]
					count10 = count10 + 1

		rand1[0] = newCentroid1[0] / count1 #R
		rand2[0] = newCentroid2[0] / count2
		rand3[0] = newCentroid3[0] / count3
		rand4[0] = newCentroid4[0] / count4
		rand5[0] = newCentroid5[0] / count5
		rand6[0] = newCentroid6[0] / count6
		rand7[0] = newCentroid7[0] / count7
		rand8[0] = newCentroid8[0] / count8
		rand9[0] = newCentroid9[0] / count9
		rand10[0] = newCentroid10[0] / count10

		rand1[1] = newCentroid1[1] / count1 #G
		rand2[1] = newCentroid2[1] / count2
		rand3[1] = newCentroid3[1] / count3
		rand4[1] = newCentroid4[1] / count4
		rand5[1] = newCentroid5[1] / count5
		rand6[1] = newCentroid6[1] / count6
		rand7[1] = newCentroid7[1] / count7
		rand8[1] = newCentroid8[1] / count8
		rand9[1] = newCentroid9[1] / count9
		rand10[1] = newCentroid10[1] / count10

		rand1[2] = newCentroid1[2] / count1 #B
		rand2[2] = newCentroid2[2] / count2
		rand3[2] = newCentroid3[2] / count3
		rand4[2] = newCentroid4[2] / count4
		rand5[2] = newCentroid5[2] / count5
		rand6[2] = newCentroid6[2] / count6
		rand7[2] = newCentroid7[2] / count7
		rand8[2] = newCentroid8[2] / count8
		rand9[2] = newCentroid9[2] / count9
		rand10[2] = newCentroid10[2] / count10


		print rand1[0], rand1[1], rand1[2]		

		if (abs(oldCentroid1[0] - rand1[0]) == 0 and abs(oldCentroid1[1] - rand1[1]) == 0 and abs(oldCentroid1[2] - rand1[2]) == 0 and abs(oldCentroid2[0] - rand2[0]) == 0 and abs(oldCentroid2[1] - rand2[1]) == 0 and abs(oldCentroid2[2] - rand2[2]) == 0 and abs(oldCentroid3[0] - rand3[0]) == 0 and abs(oldCentroid3[1] - rand3[1]) == 0 and abs(oldCentroid3[2] - rand3[2]) == 0 and abs(oldCentroid4[0] - rand4[0]) == 0 and abs(oldCentroid4[1] - rand4[1]) == 0 and abs(oldCentroid4[2] - rand4[2]) == 0 and abs(oldCentroid5[0] - rand5[0]) == 0 and abs(oldCentroid5[1] - rand5[1]) == 0 and abs(oldCentroid5[2] - rand5[2]) == 0 and abs(oldCentroid6[0] - rand6[0]) == 0 and abs(oldCentroid6[1] - rand6[1]) == 0 and abs(oldCentroid6[2] - rand6[2]) == 0 and abs(oldCentroid7[0] - rand7[0]) == 0 and abs(oldCentroid7[1] - rand7[1]) == 0 and abs(oldCentroid7[2] - rand7[2]) == 0 and abs(oldCentroid8[0] - rand8[0]) == 0 and abs(oldCentroid8[1] - rand8[1]) == 0 and abs(oldCentroid8[2] - rand8[2]) == 0 and abs(oldCentroid9[0] - rand9[0]) == 0 and abs(oldCentroid9[1] - rand9[1]) == 0 and abs(oldCentroid9[2] - rand9[2]) == 0 and abs(oldCentroid10[0] - rand10[0]) == 0 and abs(oldCentroid10[1] - rand10[1]) == 0 and abs(oldCentroid10[2] - rand10[2]) == 0):
			break
		oldCentroid1 = [rand1[0], rand1[1], rand1[2]]
		oldCentroid2 = [rand2[0], rand2[1], rand2[2]]
		oldCentroid3 = [rand3[0], rand3[1], rand3[2]]
		oldCentroid4 = [rand4[0], rand4[1], rand4[2]]
		oldCentroid5 = [rand5[0], rand5[1], rand5[2]]
		oldCentroid6 = [rand6[0], rand6[1], rand6[2]]
		oldCentroid7 = [rand7[0], rand7[1], rand7[2]]
		oldCentroid8 = [rand8[0], rand8[1], rand8[2]]
		oldCentroid9 = [rand9[0], rand9[1], rand9[2]]
		oldCentroid10 = [rand10[0], rand10[1], rand10[2]]

	R1 = 0
	G1 = 0
	B1 = 0
	count1 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 1:
				count1 += 1
				R1 += img[i][j][0]
				G1 += img[i][j][1]
				B1 += img[i][j][2]
	R1 = R1 / count1
	G1 = G1 / count1
	B1 = B1 / count1

	R2 = 0
	G2 = 0
	B2 = 0
	count2 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 2:
				count2 += 1
				R2 += img[i][j][0]
				G2 += img[i][j][1]
				B2 += img[i][j][2]

	R2 = R2 / count2
	G2 = G2 / count2
	B2 = B2 / count2

	R3 = 0
	G3 = 0
	B3 = 0
	count3 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 3:
				count3 += 1
				R3 += img[i][j][0]
				G3 += img[i][j][1]
				B3 += img[i][j][2]

	R3 = R3 / count3
	G3 = G3 / count3
	B3 = B3 / count3

	R4 = 0
	G4 = 0
	B4 = 0
	count4 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 4:
				count4 += 1
				R4 += img[i][j][0]
				G4 += img[i][j][1]
				B4 += img[i][j][2]

	R4 = R4 / count4
	G4 = G4 / count4
	B4 = B4 / count4

	R5 = 0
	G5 = 0
	B5 = 0
	count1 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 5:
				count5 += 1
				R5 += img[i][j][0]
				G5 += img[i][j][1]
				B5 += img[i][j][2]

	R5 = R5 / count5
	G5 = G5 / count5
	B5 = B5 / count5
	
	R6 = 0
	G6 = 0
	B6 = 0
	count6 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 6:
				count6 += 1
				R6 += img[i][j][0]
				G6 += img[i][j][1]
				B6 += img[i][j][2]

	R6 = R6 / count6
	G6 = G6 / count6
	B6 = B6 / count6

	R7 = 0
	G7 = 0
	B7 = 0
	count7 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 7:
				count7 += 1
				R7 += img[i][j][0]
				G7 += img[i][j][1]
				B7 += img[i][j][2]

	R7 = R7 / count7
	G7 = G7 / count7
	B7 = B7 / count7

	R8 = 0
	G8 = 0
	B8 = 0
	count8 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 8:
				count8 += 1
				R8 += img[i][j][0]
				G8 += img[i][j][1]
				B8 += img[i][j][2]

	R8 = R8 / count8
	G8 = G8 / count8
	B8 = B8 / count8


	R9 = 0
	G9 = 0
	B9 = 0
	count9 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 9:			
				count9 += 1
				R9 += img[i][j][0]
				G9 += img[i][j][1]
				B9 += img[i][j][2]

	R9 = R9 / count9
	G9 = G9 / count9
	B9 = B9 / count9

	R10 = 0
	G10 = 0
	B10 = 0
	count10 = 0
	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 10:			
				count10 += 1
				R10 += img[i][j][0]
				G10 += img[i][j][1]
				B10 += img[i][j][2]

	R10 = R10 / count10
	G10 = G10 / count10
	B10 = B10 / count10

	for i in range(rows):
		for j in range(cols):
			if membership[i][j] == 1:
				img[i][j][0] = R1
				img[i][j][1] = G1
				img[i][j][2] = B1
			elif membership[i][j] == 2:
				img[i][j][0] = R2
				img[i][j][1] = G2
				img[i][j][2] = B2
			elif membership[i][j] == 3:
				img[i][j][0] = R3
				img[i][j][1] = G3
				img[i][j][2] = B3
			elif membership[i][j] == 4:
				img[i][j][0] = R4
				img[i][j][1] = G4
				img[i][j][2] = B4
			elif membership[i][j] == 5:
				img[i][j][0] = R5
				img[i][j][1] = G5
				img[i][j][2] = B5
			elif membership[i][j] == 6:
				img[i][j][0] = R6
				img[i][j][1] = G6
				img[i][j][2] = B6
			elif membership[i][j] == 7:
				img[i][j][0] = R7
				img[i][j][1] = G7
				img[i][j][2] = B7
			elif membership[i][j] == 8:
				img[i][j][0] = R8
				img[i][j][1] = G8
				img[i][j][2] = B8
			elif membership[i][j] == 9:
				img[i][j][0] = R9
				img[i][j][1] = G9
				img[i][j][2] = B9
			elif membership[i][j] == 10:
				img[i][j][0] = R10
				img[i][j][1] = G10
				img[i][j][2] = B10
	cv2.imwrite('Part1.png', np.array(img))


if __name__ == '__main__':
	main()