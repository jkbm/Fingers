#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from sklearn.externals import joblib
from skimage.feature import hog

import numpy as np
from matplotlib import pyplot as plt
from time import time
import tkFileDialog
from Tkinter import *


#Chosing the image
file_path_string = tkFileDialog.askopenfilename()

#Load up the classifier from the file(fingerprint.py)
clf = joblib.load("fingers.pkl")

#Reading and processing the image
im = cv2.imread(file_path_string)
im =  cv2.medianBlur(im,5)
im_bin = 5
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

tih = time()
im_gray = cv2.fastNlMeansDenoising(im_gray, None, 7, )
print "Hoged in %s s" % round(time()-tih, 3)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)

im_th = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
  		cv2.THRESH_BINARY,11,2)

ret,th1 = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
             	cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	             cv2.THRESH_BINARY,11,2)
	
h = hog(th3, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1, 1), visualise=False)
print len(h)


#Predicting the answer and formating it to display
pred = clf.predict(np.array(h, 'float64'))
prediction = pred[0].replace(' ', '')[:-4]
pred_path = 'D:\dir\imgres\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\png_txt/figs_0\%s.png'%prediction
txt = open('D:\dir\imgres\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\png_txt/figs_0\%s.txt'%prediction)
line = txt.read().split()
text_dict = {}
text_list = {line[0] : line[1], line[2] : line[3], line[4] : line[5]}

#Processing and displaying the results
print text_list
imp = cv2.imread(pred_path)

if line[1] == "M":
	sx = u"Чоловіча"
else:
	sx = u"Жіноча"

if line[3] == "L":
	pat = "left loop"
elif line[3] == "R": 
	pat = "right loop"
elif line[3] == "W": 
	pat = "whirl"
elif line[3] == "T": 
	pat = "tented arch"
elif line[3] == "A":
	pat = "arch"


titles = ['Input', 'Prediction',
  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [im, imp, th2, th3]
for i in xrange(4):
     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
     plt.title(titles[i])
     plt.xticks([]),plt.yticks([])
plt.show()

print pred

root=Tk()
label1 = Label(font='sans 20')
label1['text'] = "Стать: {0}".decode("utf-8").format(sx)
label1.pack()
label2 = Label(font='sans 20')
label2['text'] = "Iсторiя: {0}".decode("utf-8").format(line[5])
label2.pack()
label3 = Label(font='sans 20')
label3['text'] = "Класс: {0}".decode("utf-8").format(pat)
label3.pack()
root.mainloop()