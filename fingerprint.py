#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Файл обучения системы базой отпечатков и их хар-ми
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from time import time
import cPickle as pkl
from matplotlib import pyplot as plt
from PIL import Image
import glob
import os
from sklearn.svm import LinearSVC



image_list = []
text_list = []

#Считка файлов изображений и добавление их в масив
for filename in glob.glob('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/*.png'): 
    imm=Image.open(filename)
    image_list.append(filename)

#Считка файлов текстового описания отпечатков и добавление их в масив
    for root,dirs,files in os.walk('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/'):
    	filename = filename[:-4]
    	tx=open(filename+'.txt')
    	#tx=open('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/' + filename)
    	text_list.append(tx.read())
    	tx.close()

print len(text_list)
line = text_list[1].split()
text_dict = {}
for x in xrange(len(text_list)):
	line = text_list[x].split()
	text_list[x] = {line[0] : line[1], line[2] : line[3], line[4] : line[5]}

print text_list[5]

labels = []
for label in xrange(len(text_list[:99])):
	labels.append(text_list[label]["History:"])
labels = np.array(labels)
print labels



hogs = []


partimg = image_list[:99]
ti = time()

#Обработка изображений для лучшего использования
for img in partimg:
	im = cv2.imread(img)


	im =  cv2.medianBlur(im,5)
	im_bin = 5
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	tih = time()
	im_gray = cv2.fastNlMeansDenoising(im_gray, None, 7, )
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
	hogs.append(h)
	 


print "Data created in %ds" % round(time()-ti, 3)

cv2.imshow("Start", im)
cv2.waitKey()

hogf = np.array(hogs, 'float64')

#Creating and training the classifier
clf = LinearSVC()
clf.fit(hogf, labels)

#Saving to file
#joblib.dump(clf, "fingers00.pkl", compress=3)
with open("fingers000.pkl", 'wb') as f : pkl.dump(clf, f)


titles = ['Original Image', 'Global Thresholding (v = 127)',
  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [im, th1, th2, th3]
for i in xrange(4):
     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
     plt.title(titles[i])
     plt.xticks([]),plt.yticks([])
plt.show()


one, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Start", th3)
cv2.waitKey()