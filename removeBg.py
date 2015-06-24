#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile
from os import mkdir, remove, listdir
from glob import glob
import multiprocessing
from scipy.misc import imread, imsave, imresize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from datetime import datetime

class Configure(object):
	def __init__(self, input_folder, output_folder, VERBOSE=False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.VERBOSE = VERBOSE

def get_classes(datasetpath):
	classes_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
	classes_paths.sort()
	classes = [basename(class_path) for class_path in classes_paths]
	return classes

def get_imgfiles(path):
	all_files = []
	all_files.extend([join(path, basename(fname))
					  for fname in glob(path + "/*")
					  if splitext(fname)[-1].lower() in [".jpg", ".jpeg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]])
	return all_files

def get_all_images(classes, conf):
	all_images = []
	print str(datetime.now())+" starting"
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		"""
		pool = multiprocessing.Pool(processes=4)
		result = [pool.apply_async(remove, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		"""
		for imName, img in enumerate(imgs):
			remove(imName, img, imageclass, conf)
		print "done "+str(imageclass)
	print str(datetime.now())+" completely done"


def trim(im):
    bg = Image.new(im.mode, im.size, (255,255,255))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -150)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def remove(imName, img, imageclass, conf):
	imName = imName+1
	im = imread(img)
	if not isdir(conf.output_folder+imageclass):
		try:
			mkdir(conf.output_folder+imageclass)
		except:
			pass
	if not isfile(conf.output_folder+imageclass+"/"+str(imName)+"_bgrem.jpg"):
		x, y, z = im.shape
		binary_im = np.empty([x,y],np.uint8)
		for i in range(0,x):
			for j in range(0,y):
				if im[i,j,1] > im[i,j,0]*1.05 and im[i,j,1] > im[i,j,2]*1.05:
					im[i,j,:] = 255
					binary_im[i,j] = 0
				else:
					binary_im[i,j] = 1
		#im = trim(Image.fromarray(im))
		plt.imshow(binary_im)
		plt.show()
		imsave(conf.output_folder+imageclass+"/"+str(imName)+"_bgrem.jpg", im)
	return str(imName)

if __name__ == "__main__":
	input_folder = "../training_2014_09_20/"
	output_folder = "../output-bg/"
	conf = Configure(input_folder, output_folder, VERBOSE=False)
	classes = get_classes(input_folder)
	get_all_images(classes, conf)