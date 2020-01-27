import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import math
from random import randint
import sys

#need to make 2 sep functions. One for img and one for the 2d array. Then we can take the determinant of the hessian

def sobelxfun(img, rows, col):
    sobelx = [[0 for x in range(col)] for y in range(rows)]
    for i in range(rows):
        for j in range(col):
            if i - 1 >= 0 and j - 1 >= 0 and i + 1 < rows and j + 1 < col: #reg case
                left = img[i][j - 1] * 2
                topLeft = img[i - 1][j - 1]* 1
                top = img[i - 1][j] * 0
                topRight = img[i - 1][j + 1] * -1
                right = img[i][j + 1] * -2
                botRight = img[i + 1][j + 1] * -1
                bot = img[i + 1][j] * 0
                botLeft = img[i + 1][j - 1] * 1
                sum = left + topLeft + top + topRight + right + botRight + bot + botLeft
                sobelx[i][j] = abs(sum)
    return sobelx

def nonmax(grad, rows, col):
    new = [[0 for x in range(col)] for y in range(rows)]
    for i in range(rows):
        for j in range(col):
            ang = math.atan(grad[i][j])
            if i - 1 >= 0 and j - 1 >= 0 and i + 1 < rows and j + 1 < col: #reg case
                if (ang > (15 * math.pi) / 8 and ang < (math.pi / 8)) or (ang > (7 * math.pi) / 8 and ang < (9 * math.pi) / 8):
                    if max(grad[i - 1][j], grad[i + 1][j], grad[i][j]) == grad[i][j]:
                        new[i][j] = grad[i][j]
                    else:
                        new[i][j] = 0
                elif (ang > (3 * math.pi) / 8 and ang < (5 * math.pi) / 8) or ((ang > (11 * math.pi) / 8) and ang < (13 * math.pi) / 8):
                    if max(grad[i][j - 1], grad[i][j + 1], grad[i][j]) == grad[i][j]:
                        new[i][j] = grad[i][j]
                    else:
                        new[i][j] = 0
                elif (ang > math.pi / 8 and ang < (3 * math.pi) / 8) or ((ang > (9 * math.pi) / 8) and ang < (11 * math.pi) / 8):
                    if max(grad[i - 1][j - 1], grad[i + 1][j + 1], grad[i][j]) == grad[i][j]:
                        new[i][j] = grad[i][j]
                    else:
                        new[i][j] = 0
                elif (ang > (5 * math.pi) / 8 and ang < (7 * math.pi) / 8) or ((ang > (13 * math.pi) / 8) and ang < (15 * math.pi)) / 8:
                    if max(grad[i - 1][j + 1], grad[i + 1][j - 1], grad[i][j]) == grad[i][j]:
                        new[i][j] = grad[i][j]
                    else:
                        new[i][j] = 0
    return new
                
    

def sobelyfun(img, rows, col):
    sobely = [[0 for x in range(col)] for y in range(rows)]
    for i in range(rows):
        for j in range(col):
            if i - 1 >= 0 and j - 1 >= 0 and i + 1 < rows and j + 1 < col: #reg case
                left = img[i][j - 1] * 0
                topLeft = img[i - 1][j - 1] * 1
                top = img[i - 1][j] * 2
                topRight = img[i - 1][j + 1] * 1
                right = img[i][j + 1] * 0
                botRight = img[i + 1][j + 1] * -1
                bot = img[i + 1][j] * -2
                botLeft = img[i + 1][j - 1] * -1
                sum = left + topLeft + top + topRight + right + botRight + bot + botLeft
                sobely[i][j] = abs(sum)
    return sobely


def ransac(points, iters, threshDis, threshPoints, rows, col):
    bool = True
    bestError = sys.float_info.max
    finalinliers = [[0 for x in range(2)] for y in range(rows * col)]
    ran1 = (0, 0) #rand for passing to output
    ran2 = (0, 0)
    while bool or bestError == sys.float_info.max:
        bool = False
        rand1posi = 0
        rand1posj = 0
        rand2posi = 0
        rand2posj = 0
        while True:
            rand1posi = randint(0, rows - 1)
            rand1posj = randint(0, col - 1)
            if points[rand1posi][rand1posj] > 0:
                break
        while True:
            rand2posi = randint(0, rows - 1)
            rand2posj = randint(0, col - 1)
            if points[rand2posi][rand2posj] > 0:
                break

            
        currerror = 0 #total distance of inliers
        currinliers = 0 #number of current inliers for the model
        inliers = [[0 for x in range(2)] for y in range(rows * col)] #collection of the inliers
        incount = 0 #keeps track of which inlier were on and stores the location of them
        for i in range(rows):
            for j in range(col):
                if points[i][j] > 0: #Changed from 0
                    p1 = np.array([i, j]) #certain point
                    p2 = np.array([rand1posi, rand1posj]) #line pt 1
                    p3 = np.array([rand2posi, rand2posj]) #line pt 2
                    #dist = abs((p3[1] - p2[1]) * p1[0] - (p3[0] - p2[0]) * p1[1] + p3[0] * p2[1]
                    #           - p3[1] * p2[0]) / math.sqrt((p3[1] - p2[1]) ** 2 +
                    #                                        (p3[0] - p2[0]) ** 2)
                    dist = abs((p3[0] - p2[0]) * p1[1] - (p3[1] - p2[1]) * p1[0] + p3[1] * p2[0]
                               - p3[0] * p2[1]) / math.sqrt((p3[0] - p2[0]) ** 2 +
                                                            (p3[1] - p2[1]) ** 2)
                    if dist <= threshDis:
                        currinliers = currinliers + 1 #inc total num of inliers
                        inliers[incount] = [i, j] #stores the position of the inlier
                        incount = incount + 1 #keeps track of position of inlier
                        currerror = currerror + dist #keeps track of total error
        if currinliers >= threshPoints and currerror < bestError:
            bestError = currerror
            finalinliers = inliers
            ran1 = (rand1posj, rand1posi) #WHY DID THEY HAVE TO BE FLIPPED
            ran2 = (rand2posj, rand2posi)
    if bestError == sys.float_info.max:
        print('No good model found for this set of iterations')
    return (finalinliers, ran1, ran2)
            

def hough(img, diag, rows, col): #trig functions take in radians and return radians
    accum = [[0 for x in range(180)] for y in range(2 * diag)] #col are theta rows are roh
    for i in range(rows): #0 to diag - 1(non inclusive) pos: diag to diag - 1 neg
        for j in range(col):
            if img[i][j] > 0: #if it is a feature point
                for theta in range(0, 180):
                    roh = int(round(j * math.cos(theta * (math.pi / 180)) + i * math.sin (theta * (math.pi / 180)))) # changed i and j
                    if roh < 0:
                        accum[diag + abs(roh)][theta] += 10 #made each vote 10 so that it is easier to see on image
                    elif roh >= 0:
                        accum[diag - roh][theta] += 10 #diag == 0; diag - roh = row
    cv2.imwrite('Hough.png', np.array(accum))

    #MAX1
    max1 = 0
    point1 = (0, 0)
    for row in range (2 * diag):
        for cols in range(180):
            if row - 1 >= 0 and cols - 1 >= 0 and row + 1 < 2 * diag and cols + 1 < 180 and accum[row][cols] > 0 and accum[row][cols] > accum[row][cols - 1] and accum[row][cols] > accum[row - 1][cols - 1] and accum[row][cols] > accum[row - 1][cols] and accum[row][cols] > accum[row - 1][cols + 1] and accum[row][cols] > accum[row][cols + 1] and accum[row][cols] > accum[row + 1][cols + 1] and accum[row][cols] > accum[row + 1][cols] and accum[row][cols] > accum[row + 1][cols - 1] and accum[row][cols] > max1: #4 parts: 1 does it have 8 neighbors, 2: is it > 0, 3: is it greater than those neighbors, 4: is it greater than the last max
                max1 = accum[row][cols] #row/theta
                point1 = (row, cols)
    
    accum[point1[0]][point1[1] - 1] = 0
    accum[point1[0] - 1][point1[1] - 1] = 0
    accum[point1[0] - 1][point1[1]] = 0
    accum[point1[0] - 1][point1[1] + 1] = 0
    accum[point1[0]][point1[1] + 1] = 0
    accum[point1[0] + 1][point1[1] + 1] = 0
    accum[point1[0] + 1][point1[1]] = 0
    accum[point1[0] + 1][point1[1] - 1] = 0
    roh1 = 0
    theta1 = point1[1]
    if point1[0] <= diag:#transfers row to rho
        roh1 = diag - point1[0]
    else:
        roh1 = -1 * (point1[0] - diag)

    a = np.cos(theta1 * (math.pi / 180))
    b = np.sin(theta1 * (math.pi / 180))
    x0 = int(round(a * roh1)) #gets point in img space and gets two other points on the line
    y0 = int(round(b * roh1))
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 =int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (y1, x1), (y2, x2), 255, 2)
    accum[point1[0]][point1[1]] = 0 #gets rid of max so that it isnt chosen again
    
    #MAX2
    max2 = 0
    point2 = (0, 0)
    for row in range (2 * diag):
        for cols in range(180):
            if row - 1 >= 0 and cols - 1 >= 0 and row + 1 < 2 * diag and cols + 1 < 180 and accum[row][cols] > 0 and accum[row][cols] > accum[row][cols - 1] and accum[row][cols] > accum[row - 1][cols - 1] and accum[row][cols] > accum[row - 1][cols] and accum[row][cols] > accum[row - 1][cols + 1] and accum[row][cols] > accum[row][cols + 1] and accum[row][cols] > accum[row + 1][cols + 1] and accum[row][cols] > accum[row + 1][cols] and accum[row][cols] > accum[row + 1][cols - 1] and accum[row][cols] > max2: #4 parts: 1 does it have 8 neighbors, 2: is it > 0, 3: is it greater than those neighbors, 4: is it greater than the last max
                max2 = accum[row][cols] #row/theta
                point2 = (row, cols)


    accum[point2[0]][point2[1] - 1] = 0
    accum[point2[0] - 1][point2[1] - 1] = 0
    accum[point2[0] - 1][point2[1]] = 0
    accum[point2[0] - 1][point2[1] + 1] = 0
    accum[point2[0]][point2[1] + 1] = 0
    accum[point2[0] + 1][point2[1] + 1] = 0
    accum[point2[0] + 1][point2[1]] = 0
    accum[point2[0] + 1][point2[1] - 1] = 0
    roh2 = 0
    theta2 = point2[1]
    if point2[0] <= diag:#transfers row to rho
        roh2 = diag - point2[0]
    else:
        roh2 = -1 * (point2[0] - diag)

    a = np.cos(theta2 * (math.pi / 180))
    b = np.sin(theta2 * (math.pi / 180))
    x0 = int(round(a * roh2))
    y0 = int(round(b * roh2))
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 =int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (y1, x1), (y2, x2), 255, 2)
    accum[point2[0]][point2[1]] = 0
    

    #MAX3
    max3 = 0
    point3 = (0, 0)
    for row in range (2 * diag):
        for cols in range(180):
            if row - 1 >= 0 and cols - 1 >= 0 and row + 1 < 2 * diag and cols + 1 < 180 and accum[row][cols] > 0 and accum[row][cols] > accum[row][cols - 1] and accum[row][cols] > accum[row - 1][cols - 1] and accum[row][cols] > accum[row - 1][cols] and accum[row][cols] > accum[row - 1][cols + 1] and accum[row][cols] > accum[row][cols + 1] and accum[row][cols] > accum[row + 1][cols + 1] and accum[row][cols] > accum[row + 1][cols] and accum[row][cols] > accum[row + 1][cols - 1] and accum[row][cols] > max3: #4 parts: 1 does it have 8 neighbors, 2: is it > 0, 3: is it greater than those neighbors, 4: is it greater than the last max
                max3 = accum[row][cols] #row/theta
                point3 = (row, cols)

    accum[point3[0]][point3[1] - 1] = 0
    accum[point3[0] - 1][point3[1] - 1] = 0
    accum[point3[0] - 1][point3[1]] = 0
    accum[point3[0] - 1][point3[1] + 1] = 0
    accum[point3[0]][point3[1] + 1] = 0
    accum[point3[0] + 1][point3[1] + 1] = 0
    accum[point3[0] + 1][point3[1]] = 0
    accum[point3[0] + 1][point3[1] - 1] = 0
    roh3 = 0
    theta3 = point3[1]
    if point3[0] <= diag:#transfers row to rho
        roh3 = diag - point3[0]
    else:
        roh3 = -1 * (point3[0] - diag)

    a = np.cos(theta3 * (math.pi / 180))
    b = np.sin(theta3 * (math.pi / 180))
    x0 = int(round(a * roh3))
    y0 = int(round(b * roh3))
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 =int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (y1, x1), (y2, x2), 255, 2)
    accum[point3[0]][point3[1]] = 0
    

    #MAX4
    max4 = 0
    point4 = (0, 0)
    for row in range (2 * diag):
        for cols in range(180):
            if row - 1 >= 0 and cols - 1 >= 0 and row + 1 < 2 * diag and cols + 1 < 180 and accum[row][cols] > 0 and accum[row][cols] > accum[row][cols - 1] and accum[row][cols] > accum[row - 1][cols - 1] and accum[row][cols] > accum[row - 1][cols] and accum[row][cols] > accum[row - 1][cols + 1] and accum[row][cols] > accum[row][cols + 1] and accum[row][cols] > accum[row + 1][cols + 1] and accum[row][cols] > accum[row + 1][cols] and accum[row][cols] > accum[row + 1][cols - 1] and accum[row][cols] > max4: #4 parts: 1 does it have 8 neighbors, 2: is it > 0, 3: is it greater than those neighbors, 4: is it greater than the last max
                max4 = accum[row][cols] #row/theta
                point4 = (row, cols)

    accum[point4[0]][point4[1] - 1] = 0
    accum[point4[0] - 1][point4[1] - 1] = 0
    accum[point4[0] - 1][point4[1]] = 0
    accum[point4[0] - 1][point4[1] + 1] = 0
    accum[point4[0]][point4[1] + 1] = 0
    accum[point4[0] + 1][point4[1] + 1] = 0
    accum[point4[0] + 1][point4[1]] = 0
    accum[point4[0] + 1][point4[1] - 1] = 0
    roh4 = 0
    theta4 = point4[1]
    if point4[0] <= diag:#transfers row to rho
        roh4 = diag - point4[0]
    else:
        roh4 = -1 * (point4[0] - diag)

    a = np.cos(theta4 * (math.pi / 180))
    b = np.sin(theta4 * (math.pi / 180))
    x0 = int(round(a * roh4))
    y0 = int(round(b * roh4))
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 =int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (y1, x1), (y2, x2), 255, 2)
    accum[point4[0]][point4[1]] = 0
    
    
    

    cv2.imwrite('houghtrans.png', np.array(img))
    #maybe the error is that i try all points not just nonzero vote points!!!!!
    

    


def main():
    img = cv2.imread('./road.png', 0)
    rows = len(img)
    col = len(img[0])
    img1 = [[0 for x in range(col)] for y in range(rows)] #performing gaussian
    for i in range(rows): #gaussian
        for j in range(col):
            if j - 1 >= 0 and j + 1 < col:
                gMid = 2 * img[i][j]
                gLeft = img[i][j - 1]
                gRight = img[i][j + 1]
                newInt = 1 / 4.0 * (gMid + gLeft + gRight)
                img1[i][j] = newInt
    for i in range(rows):
        for j in range(col):
            if i - 1 >= 0 and i + 1 < rows:
                gMid = 2 * img[i][j]
                gAbv = img[i - 1][j]
                gBel = img[i + 1][j]
                newInt = 1 / 4.0 * (gMid + gAbv + gBel)
                img[i][j] = newInt
    
    
    sobelx = [[0 for x in range(col)] for y in range(rows)]
    sobely = [[0 for x in range(col)] for y in range(rows)]    
    sobelxx = [[0 for x in range(col)] for y in range(rows)]
    sobelyy = [[0 for x in range(col)] for y in range(rows)]
    sobelxy = [[0 for x in range(col)] for y in range(rows)]
    sobelyx = [[0 for x in range(col)] for y in range(rows)]
    sobelx = sobelxfun(img1, rows, col)
    sobely = sobelyfun(img1, rows, col)
    #GOOD UP TO AT LEAST HERE
    
    sobelxx = sobelxfun(sobelx, rows, col)
    sobelyy = sobelyfun(sobely, rows, col)
    sobelxy = sobelyfun(sobelx, rows, col)
    sobelyx = sobelxfun(sobely, rows, col)
    hessianDet = [[0 for x in range(col)] for y in range(rows)]
    for i in range(rows):
        for j in range(col):
            hessianDet[i][j] = sobelxx[i][j] * sobelyy[i][j] - sobelyx[i][j] * sobelxy[i][j]
    hessianDet = np.array(hessianDet)                     
    for i in range(rows):
        for j in range(col):
            if hessianDet[i][j] < 120000: #thresholding hessianDet
                hessianDet[i][j] = 0 
    nonmaxdet = nonmax(hessianDet, rows, col)
    nonmaxdet1 = hessianDet
    for i in range(rows):
        for j in range(col):
             nonmaxdet1[i][j] = nonmaxdet[i][j]
    cv2.imwrite('HessianDet.png', np.array(nonmaxdet)) #END OF PART 1
    #start of RANSAC
    copynonmax = [[0 for x in range(col)] for y in range(rows)]
    for i in range(rows):
        for j in range(col):
            copynonmax[i][j] = nonmaxdet1[i][j]
    line1 = ransac(copynonmax, 15, 7.0, 50, rows, col)
            
    i = 0
    while line1[0][i] != [0, 0]:#if nonmaxdet1 is in line1 then make 3x3 square around it
        x = line1[0][i][0]
        y = line1[0][i][1]
        if x + 1 < rows and x - 1 >= 0 and y + 1 < col and y - 1 >= 0:
            nonmaxdet1[x][y - 1] = 255
            nonmaxdet1[x - 1][y - 1] = 255
            nonmaxdet1[x - 1][y] = 255
            nonmaxdet1[x - 1][y + 1] = 255
            nonmaxdet1[x][y + 1] = 255
            nonmaxdet1[x + 1][y + 1] = 255
            nonmaxdet1[x + 1][y] = 255
            nonmaxdet1[x + 1][y - 1] = 255
        i = i + 1
    i = 0
    j = 0
    while line1[0][j] != [0, 0]: # Makes sure same inliers are not reused
        copynonmax[line1[0][j][0]] [line1[0][j][1]] = 0
        j = j + 1
    line2 = ransac(copynonmax, 15, 7.0, 50, rows, col)
    while line2[0][i] != [0, 0]:#if nonmaxdet1 is in line1 then make 3x3 square around it
        x = line2[0][i][0]
        y = line2[0][i][1]
        if x + 1 < rows and x - 1 >= 0 and y + 1 < col and y - 1 >= 0:
            nonmaxdet1[x][y - 1] = 255
            nonmaxdet1[x - 1][y - 1] = 255
            nonmaxdet1[x - 1][y] = 255
            nonmaxdet1[x - 1][y + 1] = 255
            nonmaxdet1[x][y + 1] = 255
            nonmaxdet1[x + 1][y + 1] = 255
            nonmaxdet1[x + 1][y] = 255
            nonmaxdet1[x + 1][y - 1] = 255
        i = i + 1
    i = 0
    j = 0
    while line2[0][j] != [0, 0]: # Makes sure same inliers are not reused
        copynonmax[line2[0][j][0]] [line2[0][j][1]] = 0
        j = j + 1
    line3 = ransac(copynonmax, 15, 7.0, 50, rows, col)
    while line3[0][i] != [0, 0]:#if nonmaxdet1 is in line1 then make 3x3 square around it
        x = line3[0][i][0]
        y = line3[0][i][1]
        if x + 1 < rows and x - 1 >= 0 and y + 1 < col and y - 1 >= 0:
            nonmaxdet1[x][y - 1] = 255
            nonmaxdet1[x - 1][y - 1] = 255
            nonmaxdet1[x - 1][y] = 255
            nonmaxdet1[x - 1][y + 1] = 255
            nonmaxdet1[x][y + 1] = 255
            nonmaxdet1[x + 1][y + 1] = 255
            nonmaxdet1[x + 1][y] = 255
            nonmaxdet1[x + 1][y - 1] = 255
        i = i + 1
    i = 0
    j = 0
    while line3[0][j] != [0, 0]: # Makes sure same inliers are not reused
        copynonmax[line3[0][j][0]] [line3[0][j][1]] = 0
        j = j + 1
    line4 = ransac(copynonmax, 15, 7.0, 50, rows, col)
    while line4[0][i] != [0, 0]:#if nonmaxdet1 is in line1 then make 3x3 square around it
        x = line4[0][i][0]
        y = line4[0][i][1]
        if x + 1 < rows and x - 1 >= 0 and y + 1 < col and y - 1 >= 0:
            nonmaxdet1[x][y - 1] = 255
            nonmaxdet1[x - 1][y - 1] = 255
            nonmaxdet1[x - 1][y] = 255
            nonmaxdet1[x - 1][y + 1] = 255
            nonmaxdet1[x][y + 1] = 255
            nonmaxdet1[x + 1][y + 1] = 255
            nonmaxdet1[x + 1][y] = 255
            nonmaxdet1[x + 1][y - 1] = 2555
        i = i + 1
    lline1 = np.array(line1[0]) #turns arr of inliers into np array
    lline2 = np.array(line2[0])
    lline3 = np.array(line3[0])
    lline4 = np.array(line4[0])
    lline1 = lline1[~np.all(lline1 == 0, axis=1)] #gets rid of 0s in array
    lline2 = lline2[~np.all(lline2 == 0, axis=1)]
    lline3 = lline3[~np.all(lline3 == 0, axis=1)]
    lline4 = lline4[~np.all(lline4 == 0, axis=1)]
    extremelin1 = [[0 for x in range(2)] for j in range(len(lline1))]
    extremelin2 = [[0 for x in range(2)] for j in range(len(lline2))]
    extremelin3 = [[0 for x in range(2)] for j in range(len(lline3))]
    extremelin4 = [[0 for x in range(2)] for j in range(len(lline4))]
    for i in range(len(lline1)): #copies nonzero elemets into the extremelin
        for j in range(2):
            extremelin1[i][j] = lline1[i][j]
    for i in range(len(lline2)):
        for j in range(2):
            extremelin2[i][j] = lline2[i][j]
    for i in range(len(lline3)):
        for j in range(2):
            extremelin3[i][j] = lline3[i][j]
    for i in range(len(lline4)):
        for j in range(2):
            extremelin4[i][j] = lline4[i][j]
    #plots the max of those arrays
    cv2.line(nonmaxdet1, (max(extremelin1)[1], max(extremelin1)[0]), (min(extremelin1)[1], min(extremelin1)[0]), 255, 2)
    cv2.line(nonmaxdet1, (max(extremelin2)[1], max(extremelin2)[0]), (min(extremelin2)[1], min(extremelin2)[0]), 255, 2)
    cv2.line(nonmaxdet1, (max(extremelin3)[1], max(extremelin3)[0]), (min(extremelin3)[1], min(extremelin3)[0]), 255, 2)
    cv2.line(nonmaxdet1, (max(extremelin4)[1], max(extremelin4)[0]), (min(extremelin4)[1], min(extremelin4)[0]), 255, 2)
    cv2.imwrite('RANSAC.png', nonmaxdet1) #END OF PART 2
    #Start of Hough
    #Roh is length of diagonal
    diag = int (math.ceil(math.sqrt(rows ** 2 + col ** 2)))
    hough(np.array(nonmaxdet), diag, rows, col)
    
    
    
    
    


if __name__ == '__main__':
    main()