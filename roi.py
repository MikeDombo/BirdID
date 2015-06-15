#!/usr/bin/env python
from os.path import isdir, basename, splitext, join, isfile
from os import mkdir, remove
from glob import glob
import multiprocessing
from PIL import Image
from skimage.morphology import disk
from skimage.filters.rank import mean, median
import numpy as np
import matplotlib.pyplot as plt

class Configure(object):
	def __init__(self, input_folder, output_folder, VERBOSE=False):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.VERBOSE = VERBOSE

def get_classes(datasetpath):
	classes_paths = [files
					 for files in glob(datasetpath + "/*")
					 if isdir(files)]
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
	if not isdir(conf.output_folder):
		mkdir(conf.output_folder)
	for i, imageclass in enumerate(classes):
		imgs = get_imgfiles(join(conf.input_folder,imageclass))
		pool = multiprocessing.Pool(processes=4)
		result = [pool.apply_async(zoom, args=(imName, img, imageclass, conf)) for imName, img in enumerate(imgs)]
		res = [p.get() for p in result]
		print "done "+str(imageclass)
#zoom(0, get_imgfiles(join(conf.input_folder,classes[0]))[0], classes[0], conf)


def zoom(imName, img, imageclass, conf):
	imName = imName+1
	im = Image.open(img)
	if not isdir(conf.output_folder+imageclass):
		mkdir(conf.output_folder+imageclass)
	x, y = im.size
	x1=0
	y1=0
	means=[]
	while y1<=y-480:
		while x1<=x-640:
			ims = im.crop((x1, y1, x1+640, y1+480))
			mean1 = mean(np.array(ims)[:,:,0], disk(640))
			for i, a in enumerate(mean1):
				mean1[i] = np.mean(a)
			mean1 = np.mean(mean1)
			means.append(mean1)
			if (len(means)>2):
				if (abs(means[len(means)-1]-means[len(means)-2])>25):
					print "trig"
					if (x1!=0):
						print "maybe bird?"
						ims.save(join(conf.output_folder,imageclass)+"/"+str(imName)+"_zoom_"+str(x1)+"_"+str(y1)+".jpg")
			x1 = x1+160
		x1=0
		y1 = y1+120
	return str(imName)

if __name__ == "__main__":
	input_folder = "../training_2014_09_20/"
	output_folder = "../output-roi/"
	conf = Configure(input_folder, output_folder, VERBOSE=False)
	classes = get_classes(input_folder)
	get_all_images(classes, conf)